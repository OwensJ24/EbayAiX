"""FastAPI app: multi-photo upload -> Claude Vision Subagent -> eBay comps/listing."""

from __future__ import annotations

import io
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, TypeVar

import anthropic
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from src.agents.vision_subagent import ProductIdentification, VisionSubagent
from src.ebay.browse import build_query, get_application_access_token, load_ebay_browse_config, search_comparable_listings
from src.ebay.config import load_ebay_config
from src.ebay.listing import EbayTradingApiError, create_listing, get_category_aspects, resolve_draft_listing, suggest_categories
from src.ebay.seller_location import save_seller_location
from src.ebay.token_store import get_valid_access_token
from src.storage.supabase_storage import load_supabase_storage_config, public_url, upload_image
from src.web.ebay_routes import router as ebay_router

# INFO-level logs (e.g. src.ebay.listing's request/response diagnostics) are silently
# dropped otherwise — Python's root logger defaults to WARNING, and nothing else in
# this app configures logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# The whole batch a single /api/identify call accepts, potentially spanning several
# different items — the user splits them into items client-side (see index.html's
# split-divider UI) before uploading, so this is just an overall sanity cap on request
# size, not tied to any per-call Claude cost the way it was when clustering was
# AI-driven.
MAX_BATCH_IMAGES = 60
# eBay's own real per-listing limit (confirmed against eBay's docs: AddFixedPriceItem
# accepts up to 24 self-hosted PictureURLs) — enforced defensively at publish time
# (see publish_listing_route() below), not at upload time, since the user could in
# principle put more than 24 photos in one split group.
MAX_ITEM_IMAGES = 24
# Only the first MAX_ANALYSIS_IMAGES of *one item's* photos are sent to Claude's
# identify() call (the deep, per-item analysis — see /api/identify/confirm below),
# independent of how many photos that item ends up with — a handful of angles is
# enough to identify an item, and analyzing all of them would multiply vision-call
# cost for little added benefit.
MAX_ANALYSIS_IMAGES = 5
# preview() (the quick name+category guess run for every item right after upload —
# see /api/identify below) needs far less than identify() does; fewer, smaller images
# keeps that pass fast, which matters a lot more here since it runs once per item,
# blocking the menu screen from appearing at all until every item's guess is back.
PREVIEW_ANALYSIS_IMAGES = 2
# Longest side, in pixels, for the extra-small thumbnails built just for preview()
# calls (see _thumbnail_for_preview() below) — distinct from EBAY_MAX_IMAGE_DIMENSION,
# since a quick name+category guess doesn't need eBay-listing quality.
PREVIEW_THUMBNAIL_SIZE = 512
# Caps how many Supabase uploads / Claude preview() calls run at once in
# /api/identify's two loops (see below) — high enough to collapse "sum of N/M
# latencies" down to roughly "max single latency" for realistic batch sizes, capped
# so a very large batch doesn't fire an unbounded number of simultaneous requests at
# either API.
_UPLOAD_CONCURRENCY = 8
# eBay's own image requirements: below 500px on either dimension, "the listing may be
# blocked" — silently, not with a clean rejection at listing-creation time, which is
# exactly the "image not showing up" failure mode this guards against.
EBAY_MIN_IMAGE_DIMENSION = 500
# eBay's own docs say "preferably at least 1600 by 1600 pixels" — not a hard minimum,
# just a quality preference — so downscaling anything larger than this to 1600px on its
# longest side stays within eBay's own guidance while capping how much raw pixel data
# gets base64-encoded for each Claude vision call and uploaded to Supabase/eBay. A
# full-resolution modern phone photo (4000px+, several MB), multiplied by up to
# MAX_BATCH_IMAGES per upload, was plausibly large enough to push Render's free-tier request past
# its timeout or memory limit with zero application-level logging — the request dies
# mid-flight (a 502 from Render's own proxy, not one of this app's own error responses)
# before any of our own logging or error handling ever runs.
EBAY_MAX_IMAGE_DIMENSION = 1600

# Built once, reused for every color-managed conversion in _to_srgb() below.
_SRGB_PROFILE = ImageCms.createProfile("sRGB")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AgentX")
app.include_router(ebay_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Loaded once at process startup — API client construction is too expensive to redo
# per request.
_vision_subagent = VisionSubagent()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _read_upload_within_limit(file: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := file.file.read(1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {max_bytes // (1024 * 1024)}MB limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _to_srgb(image: Image.Image) -> Image.Image:
    """Color-manages the image into real sRGB pixel values using its embedded ICC
    profile (if any), rather than just tagging/dropping metadata. iPhones (and many
    Android cameras) shoot in Display P3, a wider gamut than sRGB — a plain
    `.convert("RGB")` leaves the original P3-authored numeric pixel values untouched,
    and this app's save() call never re-attached the embedded profile, so any viewer
    that assumes sRGB (which is most of them, and — critically — very likely eBay's own
    image pipeline, which we can't confirm is color-profile-aware) rendered those
    numbers as if they were already sRGB: real, saturated colors came out visibly
    duller/greyer. Converting the actual pixels to sRGB up front fixes this everywhere,
    color-managed viewer or not, since the raw bytes are correct instead of relying on
    downstream profile support. Falls back to a plain mode conversion when there's no
    embedded profile (most images, already sRGB) or the profile can't be parsed.
    """
    icc_bytes = image.info.get("icc_profile")
    if not icc_bytes:
        return image.convert("RGB")
    try:
        source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        return ImageCms.profileToProfile(image, source_profile, _SRGB_PROFILE, outputMode="RGB")
    except ImageCms.PyCMSError as e:
        logger.info("Ignoring unparseable embedded ICC profile: %r", e)
        return image.convert("RGB")


def _validate_and_normalize_image(contents: bytes) -> bytes:
    """Validates dimensions and always re-encodes to JPEG (downscaling first if needed)
    — unconditionally, not just when downscaling. This keeps every upload's Supabase
    Storage object path fully deterministic (`{upload_id}/{index}.jpg`, see
    src/storage/supabase_storage.py), so nothing downstream ever needs to track or look
    up the original file's format/extension. Minor tradeoff: already-small JPEGs get a
    redundant re-compression pass at quality=88 (visually near-lossless for product
    photos) instead of passing through untouched.
    """
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File isn't a readable image.")

    # Phone cameras store photos in raw sensor orientation plus an EXIF Orientation tag
    # telling viewers how to rotate/flip it for display — Image.open() doesn't apply
    # that automatically. Without this, a portrait phone photo gets its correct-looking
    # preview in the browser (which does honor EXIF) but silently loses that rotation
    # once re-encoded below (no EXIF carried forward), coming out sideways everywhere
    # downstream, including on the published eBay listing. exif_transpose() bakes the
    # rotation into the actual pixel data and strips the now-redundant tag, so this must
    # run before width/height are read below — a portrait photo's sensor-native
    # width/height are swapped relative to its correctly-oriented display size.
    image = ImageOps.exif_transpose(image)

    width, height = image.size
    if width < EBAY_MIN_IMAGE_DIMENSION or height < EBAY_MIN_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image is {width}x{height}px — eBay requires at least "
                f"{EBAY_MIN_IMAGE_DIMENSION}x{EBAY_MIN_IMAGE_DIMENSION}px "
                "(smaller images may be silently blocked from listings). Upload a larger photo."
            ),
        )

    image = _to_srgb(image)
    if max(width, height) > EBAY_MAX_IMAGE_DIMENSION:
        image.thumbnail((EBAY_MAX_IMAGE_DIMENSION, EBAY_MAX_IMAGE_DIMENSION), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _read_and_normalize_upload(file: UploadFile) -> bytes:
    """Reads, validates, and normalizes one uploaded file to JPEG bytes — entirely
    in-memory. No local disk staging: the only thing that ever needed a real file path
    was the local ResNet50 classifier, which this app no longer has, so photos go
    straight from upload to Supabase (see /api/identify below)."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'unknown'}")

    contents = _read_upload_within_limit(file, MAX_UPLOAD_BYTES)
    return _validate_and_normalize_image(contents)


def _thumbnail_for_preview(image_bytes: bytes) -> bytes:
    """Downscales an already-normalized (eBay-quality, up to 1600px) image to a small
    thumbnail just for preview() calls — a quick name+category guess doesn't need
    eBay-listing quality, and smaller images mean faster per-call vision processing.
    The original normalized bytes (not this thumbnail) are what's uploaded to Supabase
    and later sent to identify() for the real per-item analysis."""
    image = Image.open(io.BytesIO(image_bytes))
    image.thumbnail((PREVIEW_THUMBNAIL_SIZE, PREVIEW_THUMBNAIL_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def _call_claude(fn: Callable[..., T], *args, **kwargs) -> T:
    """Run a Claude-backed subagent call, translating transient API failures into clean HTTP errors."""
    try:
        return fn(*args, **kwargs)
    except anthropic.OverloadedError:
        raise HTTPException(
            status_code=503,
            detail="Claude's API is temporarily overloaded. Please try again in a moment.",
        )
    except anthropic.RateLimitError:
        raise HTTPException(status_code=503, detail="Rate limit reached. Please try again shortly.")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=503, detail="Could not reach Claude's API. Check your network connection.")
    except anthropic.APIResponseValidationError as e:
        # Raised when Claude's response doesn't validate against the requested
        # structured-output schema (output_format=...) — a real, if uncommon, failure
        # mode distinct from APIStatusError (this is APIError's other direct subclass,
        # so it wasn't caught by the clause below until this was added). Logged with
        # full detail since this is otherwise a silent, untraceable failure.
        logger.exception("Claude structured-output response failed schema validation")
        raise HTTPException(
            status_code=502,
            detail=f"Claude returned a response that didn't match the expected format ({e}). Please try again.",
        )
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e.message}")
    except Exception as e:
        # Last-resort net: any other exception from a Claude call (e.g. a raw
        # pydantic.ValidationError) used to propagate as an opaque, untraceable 500.
        # Logging it here means the next occurrence is actually diagnosable via logs.
        logger.exception("Unexpected error during a Claude-backed call")
        raise HTTPException(status_code=502, detail=f"Unexpected error calling Claude: {type(e).__name__}: {e}")


def _call_storage(fn: Callable[..., T], *args, **kwargs) -> T:
    """Run a Supabase Storage-backed call, translating failures into clean HTTP errors."""
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        # load_supabase_storage_config() raises this for missing env vars.
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502, detail=f"Supabase Storage error: {e.response.status_code} {e.response.text[:300]}"
        )
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="Could not reach Supabase Storage. Please try again.")


def _fetch_image(url: str) -> bytes:
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    return response.content


@app.post("/api/identify")
def identify(files: list[UploadFile] = File(...), group_sizes: str = Form(...)) -> dict:
    """Uploads a whole batch of photos — potentially several different items in one
    session — already split into per-item groups by the user (the frontend's
    split-divider UI, see index.html: photos load in a sequential strip and the user
    taps between two photos to mark an item boundary; automatic Claude-based clustering
    was tried first and replaced per explicit direction — it didn't group reliably
    enough to trust). Never runs the full analysis; the user picks one item at a time
    from the resulting menu and confirms/edits its name+category via
    POST /api/identify/confirm before brand/model/condition/description get analyzed
    at all (see that route below).

    `group_sizes` is a JSON-encoded array of positive integers (e.g. "[3, 2, 4]")
    giving each item's photo count, in the same order as `files` — since splits only
    ever divide the sequential upload order, not reorder it, contiguous ranges are
    always enough to describe a group; there's no need for arbitrary index lists the
    way an AI-clustering result would have needed.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one photo is required.")
    if len(files) > MAX_BATCH_IMAGES:
        raise HTTPException(status_code=400, detail=f"Up to {MAX_BATCH_IMAGES} photos are allowed per upload.")

    try:
        sizes = json.loads(group_sizes)
    except ValueError:
        raise HTTPException(status_code=400, detail="group_sizes must be valid JSON.")
    if not isinstance(sizes, list) or not sizes or any(not isinstance(s, int) or s < 1 for s in sizes):
        raise HTTPException(status_code=400, detail="group_sizes must be a non-empty JSON array of positive integers.")
    if sum(sizes) != len(files):
        raise HTTPException(
            status_code=400,
            detail=f"group_sizes sums to {sum(sizes)}, but {len(files)} photos were uploaded.",
        )

    # Validate/normalize every file into memory *before* uploading any of them, so a
    # single bad file (e.g. too small) aborts cleanly with nothing partially written to
    # Supabase.
    normalized_images = [_read_and_normalize_upload(f) for f in files]

    upload_id = uuid.uuid4().hex
    storage_config = _call_storage(load_supabase_storage_config)
    # Uploaded to Supabase Storage immediately — this, not this app's own server, is
    # what eBay will fetch imageUrls from later (see src/storage/supabase_storage.py
    # for why: Render's free tier sleeps, but Supabase doesn't, and eBay fetches images
    # lazily rather than synchronously). Run concurrently, not one at a time — each
    # upload is an independent network round trip to its own {upload_id}/{index}.jpg
    # path, so there's no ordering or shared-state reason to serialize them, and doing
    # so was directly responsible for a large chunk of this route's latency on batches
    # with many photos.
    with ThreadPoolExecutor(max_workers=_UPLOAD_CONCURRENCY) as executor:
        futures = [
            executor.submit(_call_storage, upload_image, storage_config, upload_id, i, image_bytes)
            for i, image_bytes in enumerate(normalized_images)
        ]
        for future in as_completed(futures):
            future.result()  # re-raises any HTTPException _call_storage produced

    # One cheap preview() call per item (not one big call over the whole batch, now
    # that grouping is the user's own manual input rather than something Claude has to
    # figure out) — gives each menu card a real name/category guess instead of a
    # placeholder. Also run concurrently: this loop previously ran every item's
    # preview() call one after another, so wall-clock time scaled directly with item
    # count even though each individual call is meant to be a "quick" guess — the
    # biggest single contributor to this route's overall latency. Each call also now
    # gets only PREVIEW_ANALYSIS_IMAGES small thumbnails (see _thumbnail_for_preview())
    # instead of MAX_ANALYSIS_IMAGES full eBay-quality photos, since a name+category
    # guess doesn't need either the extra images or the extra resolution — that's
    # what identify() (the real per-item analysis, run later, only for items the user
    # actually opens) is for.
    def _preview_item(photo_indices: list[int]) -> dict:
        analysis_images = [
            (_thumbnail_for_preview(normalized_images[i]), "image/jpeg")
            for i in photo_indices[:PREVIEW_ANALYSIS_IMAGES]
        ]
        preview = _call_claude(_vision_subagent.preview, analysis_images)
        return {"photo_indices": photo_indices, "item_name": preview.item_name, "category": preview.category}

    item_photo_indices = []
    offset = 0
    for size in sizes:
        item_photo_indices.append(list(range(offset, offset + size)))
        offset += size

    with ThreadPoolExecutor(max_workers=_UPLOAD_CONCURRENCY) as executor:
        # dict, not list(executor.map(...)), so results land back in split order even
        # though they may finish out of order — executor.map would preserve order too,
        # but future-per-item keeps this loop's error handling identical in shape to
        # the upload loop above rather than mixing two different concurrency idioms.
        future_to_position = {
            executor.submit(_preview_item, photo_indices): position
            for position, photo_indices in enumerate(item_photo_indices)
        }
        items_by_position: dict[int, dict] = {}
        for future in as_completed(future_to_position):
            items_by_position[future_to_position[future]] = future.result()
        items = [items_by_position[i] for i in range(len(item_photo_indices))]

    return {
        "status": "batch_preview",
        "upload_id": upload_id,
        "image_count": len(files),
        "items": [{"item_index": idx, **item} for idx, item in enumerate(items)],
    }


class CategorySearchRequest(BaseModel):
    query: str


@app.post("/api/categories/search")
def search_categories(payload: CategorySearchRequest) -> dict:
    """Lets the confirm-item screen re-search when none of the initial category
    suggestions fit, using different search terms than the LLM's own guess."""
    browse_config = _call_ebay(load_ebay_browse_config)
    app_token = _call_ebay(get_application_access_token, browse_config)
    suggestions = _call_ebay(suggest_categories, browse_config, app_token, payload.query)
    return {"status": "complete", "result": {"suggestions": [s.model_dump() for s in suggestions]}}


class ConfirmItemRequest(BaseModel):
    upload_id: str
    item_name: str
    category_id: str
    category_name: str
    photo_indices: list[int]


@app.post("/api/identify/confirm")
def confirm_item(payload: ConfirmItemRequest) -> dict:
    """Phase 2 of 2: the full Claude analysis (brand/model/condition/description),
    run only after the user has confirmed a real item name + eBay category — those are
    passed in as ground truth context rather than re-guessed.

    This is a separate HTTP request from /api/identify with no server-side session
    state, so the first MAX_ANALYSIS_IMAGES of this item's own photos (its
    photo_indices, as assigned by /api/identify's clustering — not necessarily a
    contiguous range, since a batch can hold several items' photos interleaved) are
    re-fetched from their already-uploaded Supabase public URLs rather than kept in
    memory across requests.
    """
    storage_config = _call_storage(load_supabase_storage_config)
    analysis_images = [
        (_call_storage(_fetch_image, public_url(storage_config, payload.upload_id, i)), "image/jpeg")
        for i in payload.photo_indices[:MAX_ANALYSIS_IMAGES]
    ]

    # Application-level eBay token — needed to fetch this category's item specifics
    # (both required and recommended — see get_category_aspects()) so identify() can
    # actively look for each one across all photos (e.g. a size printed on a tag),
    # rather than relying purely on post-hoc text matching in listing.py's
    # resolve_aspects().
    browse_config = _call_ebay(load_ebay_browse_config)
    app_token = _call_ebay(get_application_access_token, browse_config)
    category_aspects = _call_ebay(get_category_aspects, browse_config, app_token, payload.category_id)

    identification = _call_claude(
        _vision_subagent.identify,
        analysis_images,
        payload.item_name,
        payload.category_name,
        category_aspects,
    )

    identification.category_id = payload.category_id
    return {"status": "complete", "upload_id": payload.upload_id, "result": identification.model_dump()}


def _call_ebay(fn: Callable[..., T], *args, **kwargs) -> T:
    """Run an eBay-backed call, translating failures into clean HTTP errors."""
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        # load_ebay_config()/load_ebay_browse_config()/get_valid_access_token() raise
        # this for missing env vars or a not-yet-connected eBay account.
        raise HTTPException(status_code=400, detail=str(e))
    except EbayTradingApiError as e:
        # Trading API calls (AddFixedPriceItem) return HTTP 200 even on application
        # failure — the real error signal is Ack=Failure in the body, which
        # listing.py's _call_trading_api() already parses into this exception.
        raise HTTPException(status_code=502, detail=f"eBay API error: {e}")
    except httpx.HTTPStatusError as e:
        request_id = (
            e.response.headers.get("X-EBAY-C-REQUEST-ID")
            or e.response.headers.get("x-ebay-request-id")
            or e.response.headers.get("rlogid")
        )
        detail = f"eBay API error: {e.response.status_code} {e.response.text[:300]}"
        if request_id:
            detail += f" (eBay request id: {request_id})"
        raise HTTPException(status_code=502, detail=detail)
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="Could not reach eBay's API. Please try again.")


class SellerLocationRequest(BaseModel):
    country: str = "US"
    postal_code: str


@app.post("/api/ebay/location")
def save_seller_location_route(payload: SellerLocationRequest) -> dict:
    """One-time account setup: saves the seller's ship-from country/postal code
    locally. Trading API's AddFixedPriceItem needs only these two flat fields on the
    item (Item.Country/Item.PostalCode) — unlike the old Inventory API, there's no
    eBay-side location object to register at all, so this never calls eBay."""
    save_seller_location(payload.country, payload.postal_code)
    return {"status": "complete"}


@app.post("/api/price")
def price(identification: ProductIdentification) -> dict:
    query = build_query(identification)
    comps = _call_ebay(search_comparable_listings, query)
    return {
        "status": "complete",
        "result": {"query": query, "comparable_listings": [c.model_dump() for c in comps]},
    }


class DraftListingRequest(BaseModel):
    identification: ProductIdentification
    upload_id: str


@app.post("/api/listing/draft")
def resolve_draft_listing_route(payload: DraftListingRequest) -> dict:
    """Resolves category/condition/aspects/policies and reports what's included vs.
    missing — a read-only dry run, not a real eBay write. Trading API's
    AddFixedPriceItem has no draft/unpublished state (see src/ebay/listing.py's module
    docstring), so nothing is created on eBay until the explicit Publish click below."""
    result = _call_ebay(resolve_draft_listing, payload.identification)
    return {"status": "complete", "result": result.model_dump()}


class UpdateDraftListingRequest(BaseModel):
    identification: ProductIdentification
    upload_id: str
    category_query: str | None = None


@app.post("/api/listing/draft/{upload_id}/update")
def update_draft_listing_route(upload_id: str, payload: UpdateDraftListingRequest) -> dict:
    """Re-resolves after edits made in the review screen (title/brand/condition/
    description/category) — same read-only dry run as above, just re-run with the
    edited fields. `upload_id` in the path is purely for URL readability; nothing is
    looked up by it since there's no eBay object to reference before publish."""
    result = _call_ebay(resolve_draft_listing, payload.identification, payload.category_query)
    return {"status": "complete", "result": result.model_dump()}


class PublishListingRequest(BaseModel):
    identification: ProductIdentification
    price: float
    weight_lbs: float
    currency: str = "USD"
    photo_indices: list[int]
    # Since several items can share one upload_id (a batch), item_index disambiguates
    # this item's SKU from any other item published from the same batch — see
    # generate_sku() in listing.py.
    item_index: int


@app.post("/api/listing/publish/{upload_id}")
def publish_listing_route(upload_id: str, payload: PublishListingRequest) -> dict:
    """Make a real, live, publicly purchasable eBay listing — the one and only eBay
    write in the whole flow (see create_listing()'s docstring). This app's own
    listing-preview screen is the Human-in-the-Loop review step, since eBay has no
    unpublished state to review beforehand; this route is only ever called from an
    explicit user click after that review, never automatically."""
    storage_config = _call_storage(load_supabase_storage_config)
    # [:MAX_ITEM_IMAGES] is a defensive cap, not an expected trim — a cluster could in
    # principle end up with more than eBay's 24-photo-per-listing limit if a batch is
    # mostly one item, and this guarantees publish never exceeds it regardless.
    image_urls = [public_url(storage_config, upload_id, i) for i in payload.photo_indices[:MAX_ITEM_IMAGES]]
    result = _call_ebay(
        create_listing,
        payload.identification,
        upload_id,
        payload.item_index,
        image_urls,
        payload.price,
        payload.weight_lbs,
        payload.currency,
    )
    # Deliberately NOT deleting the Supabase image here, even though eBay has by this
    # point confirmed the listing is live. eBay's image fetch is lazy/asynchronous (the
    # same reasoning behind the original "image not available" bug), so there's no
    # reliable point at which deleting the source image is provably safe. Leaving it in
    # Supabase permanently is the same accepted tradeoff this codebase already makes for
    # abandoned local uploads — accumulating storage is a much smaller problem than a
    # real, live, published listing showing a broken image.
    return {"status": "complete", "result": result.model_dump()}
