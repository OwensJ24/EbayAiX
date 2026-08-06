"""FastAPI app: multi-photo upload -> Claude Vision Subagent -> eBay comps/listing."""

from __future__ import annotations

import io
import logging
import uuid
from pathlib import Path
from typing import Callable, TypeVar

import anthropic
import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageCms, UnidentifiedImageError
from pydantic import BaseModel

from src.agents.vision_subagent import ItemCluster, ProductIdentification, VisionSubagent
from src.ebay.browse import build_query, get_application_access_token, load_ebay_browse_config, search_comparable_listings
from src.ebay.config import load_ebay_config
from src.ebay.listing import EbayTradingApiError, create_listing, get_required_aspects, resolve_draft_listing, suggest_categories
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
# different items — Claude clusters them by item (see cluster_items() below) before
# anything else happens. Bounded because the clustering call sends every photo in the
# batch to Claude in one message; unbounded would mean unbounded cost/context per
# upload.
MAX_BATCH_IMAGES = 60
# eBay's own real per-listing limit (confirmed against eBay's docs: AddFixedPriceItem
# accepts up to 24 self-hosted PictureURLs) — enforced defensively at publish time
# (see publish_listing_route() below), not at upload time, since how many photos end
# up clustered into one item isn't known until after cluster_items() runs.
MAX_ITEM_IMAGES = 24
# Only the first MAX_ANALYSIS_IMAGES of *one item's* photos are sent to Claude's
# identify() call, independent of how many photos that item ends up with — a handful
# of angles is enough to identify an item, and analyzing all of them would multiply
# vision-call cost for little added benefit.
MAX_ANALYSIS_IMAGES = 5
# Longest side, in pixels, for the separate thumbnail set built just for the batch
# clustering call (see _thumbnail_for_clustering() below) — distinct from
# EBAY_MAX_IMAGE_DIMENSION below, since clustering only needs enough resolution to
# visually distinguish items, not eBay-listing quality, and a batch can be up to
# MAX_BATCH_IMAGES full images in one Claude message.
CLUSTERING_THUMBNAIL_SIZE = 512
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


def _thumbnail_for_clustering(image_bytes: bytes) -> bytes:
    """Downscales an already-normalized (eBay-quality, up to 1600px) image to a small
    thumbnail just for the batch clustering call — a batch can be up to MAX_BATCH_IMAGES
    full images, and clustering only needs enough resolution to visually tell items
    apart, not eBay-listing quality. The original normalized bytes (not this thumbnail)
    are what's actually uploaded to Supabase and later analyzed per-item."""
    image = Image.open(io.BytesIO(image_bytes))
    image.thumbnail((CLUSTERING_THUMBNAIL_SIZE, CLUSTERING_THUMBNAIL_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def _normalize_clusters(clusters: list[ItemCluster], total: int) -> list[ItemCluster]:
    """Validates/repairs Claude's clustering output before it's trusted: drops
    out-of-range indices, dedupes (first assignment wins), and appends any photo index
    Claude never assigned as its own singleton item — so every uploaded photo always
    ends up belonging to exactly one item, never silently dropped. Falls back to one
    cluster containing every photo if Claude's output is unusable (e.g. empty), rather
    than failing the whole upload.
    """
    seen: set[int] = set()
    repaired: list[ItemCluster] = []
    for cluster in clusters:
        indices = [i for i in cluster.photo_indices if 0 <= i < total and i not in seen]
        if not indices:
            continue
        seen.update(indices)
        repaired.append(ItemCluster(photo_indices=indices, item_name=cluster.item_name, category=cluster.category))

    missing = [i for i in range(total) if i not in seen]
    for i in missing:
        repaired.append(ItemCluster(photo_indices=[i], item_name="Unidentified Item", category="Other"))

    if not repaired:
        repaired = [ItemCluster(photo_indices=list(range(total)), item_name="Unidentified Item", category="Other")]

    return repaired


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
def identify(files: list[UploadFile] = File(...)) -> dict:
    """Uploads a whole batch of photos — potentially several different items in one
    session — and groups them by distinct physical item. Never runs the full analysis;
    the user picks one detected item at a time from the resulting menu and confirms/
    edits its name+category via POST /api/identify/confirm before brand/model/
    condition/description get analyzed at all (see that route below).

    Accepts up to MAX_BATCH_IMAGES photos total. Every photo is uploaded to Supabase
    regardless of which item it ends up clustered into (all of them get attached to
    their item's eBay listing at publish time — see src/ebay/listing.py's
    create_listing()); clustering itself runs on a separate, smaller thumbnail set
    (see _thumbnail_for_clustering()) to keep that one Claude call's cost bounded.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one photo is required.")
    if len(files) > MAX_BATCH_IMAGES:
        raise HTTPException(status_code=400, detail=f"Up to {MAX_BATCH_IMAGES} photos are allowed per upload.")

    # Validate/normalize every file into memory *before* uploading any of them, so a
    # single bad file (e.g. too small) aborts cleanly with nothing partially written to
    # Supabase.
    normalized_images = [_read_and_normalize_upload(f) for f in files]

    upload_id = uuid.uuid4().hex
    storage_config = _call_storage(load_supabase_storage_config)
    # Uploaded to Supabase Storage immediately — this, not this app's own server, is
    # what eBay will fetch imageUrls from later (see src/storage/supabase_storage.py
    # for why: Render's free tier sleeps, but Supabase doesn't, and eBay fetches images
    # lazily rather than synchronously).
    for i, image_bytes in enumerate(normalized_images):
        _call_storage(upload_image, storage_config, upload_id, i, image_bytes)

    clustering_images = [(_thumbnail_for_clustering(b), "image/jpeg") for b in normalized_images]
    cluster_result = _call_claude(_vision_subagent.cluster_items, clustering_images)
    items = _normalize_clusters(cluster_result.items, len(files))

    return {
        "status": "batch_preview",
        "upload_id": upload_id,
        "image_count": len(files),
        "items": [
            {
                "item_index": idx,
                "photo_indices": item.photo_indices,
                "item_name": item.item_name,
                "category": item.category,
            }
            for idx, item in enumerate(items)
        ],
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

    # Application-level eBay token — needed to fetch this category's required item
    # specifics so identify() can actively look for each one across all photos (e.g. a
    # size printed on a tag), rather than relying purely on post-hoc text matching in
    # listing.py's resolve_aspects().
    browse_config = _call_ebay(load_ebay_browse_config)
    app_token = _call_ebay(get_application_access_token, browse_config)
    required_aspects = _call_ebay(get_required_aspects, browse_config, app_token, payload.category_id)

    identification = _call_claude(
        _vision_subagent.identify,
        analysis_images,
        payload.item_name,
        payload.category_name,
        required_aspects,
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
