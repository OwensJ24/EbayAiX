"""FastAPI app: image upload -> local classifier -> Claude Vision Subagent -> eBay comps/listing."""

from __future__ import annotations

import gc
import io
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Callable, TypeVar

import anthropic
import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.agents.vision_subagent import ProductIdentification, VisionSubagent
from src.ebay.browse import build_query, get_application_access_token, load_ebay_browse_config, search_comparable_listings
from src.ebay.config import load_ebay_config
from src.ebay.listing import (
    create_draft_listing,
    create_inventory_location,
    get_offer,
    publish_offer,
    suggest_categories,
    update_draft_listing,
)
from src.ebay.token_store import get_valid_access_token
from src.ml.vision_preprocessor import ClassificationResult, VisionPreprocessor
from src.web.ebay_routes import router as ebay_router

# INFO-level logs (e.g. src.ebay.listing's request/response diagnostics) are silently
# dropped otherwise — Python's root logger defaults to WARNING, and nothing else in
# this app configures logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

T = TypeVar("T")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# eBay's own Inventory API image requirements: below 500px on either dimension, "the
# listing may be blocked" — silently, not with a clean rejection at listing-creation
# time, which is exactly the "image not showing up" failure mode this guards against.
EBAY_MIN_IMAGE_DIMENSION = 500

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = _PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AgentX")
app.include_router(ebay_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Loaded once at process startup — ResNet50 weight loading and API client
# construction are both too expensive to redo per request.
_preprocessor = VisionPreprocessor()
_vision_subagent = VisionSubagent()

# Serializes local ResNet50 inference across requests. Each `/api/identify` call
# runs in its own thread-pool thread (FastAPI's default for sync routes), so
# multiple images uploaded in quick succession would otherwise run CPU inference
# concurrently — each pass is memory-hungry enough that 2-3 at once can exceed a
# constrained container's memory limit (observed causing OOM crashes on Render's
# free tier). This queues them instead of running them in parallel; Claude API
# calls afterward are network-bound and stay unserialized.
_inference_lock = threading.Lock()


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


def _validate_image_dimensions(contents: bytes) -> None:
    try:
        with Image.open(io.BytesIO(contents)) as image:
            width, height = image.size
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File isn't a readable image.")

    if width < EBAY_MIN_IMAGE_DIMENSION or height < EBAY_MIN_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image is {width}x{height}px — eBay requires at least "
                f"{EBAY_MIN_IMAGE_DIMENSION}x{EBAY_MIN_IMAGE_DIMENSION}px "
                "(smaller images may be silently blocked from listings). Upload a larger photo."
            ),
        )


def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'unknown'}")

    contents = _read_upload_within_limit(file, MAX_UPLOAD_BYTES)
    _validate_image_dimensions(contents)
    upload_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{upload_id}{suffix}"
    path.write_bytes(contents)
    return path


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


@app.post("/api/identify")
def identify(file: UploadFile = File(...)) -> dict:
    """Phase 1 of 2: a cheap preview guess (name + category) only — never the full
    analysis. The user must confirm/edit both via POST /api/identify/confirm before
    brand/model/condition/description get analyzed at all (see that route below)."""
    path = _save_upload(file)
    try:
        with _inference_lock:
            local_result = _preprocessor.classify(path)
            # PyTorch's CPU allocator doesn't reliably return freed memory to the OS
            # between requests, so RSS tends to ratchet upward run over run rather than
            # reset — on Render's free-tier 512MB limit this reliably OOMs by the second
            # full identify pass. Forcing collection right after the heaviest allocation
            # (image tensor + inference) while still holding the lock (so it can't race
            # a concurrent classify()) measurably reduces that carryover.
            gc.collect()
        preview = _call_claude(_vision_subagent.preview, path, local_result)
    except Exception:
        path.unlink(missing_ok=True)
        raise

    # Application-level (client-credentials) eBay token, not the user's OAuth one —
    # category suggestions are public Taxonomy data, so this works even before the user
    # has connected their own eBay account (same reasoning as browse.py's comp pricing).
    browse_config = _call_ebay(load_ebay_browse_config)
    app_token = _call_ebay(get_application_access_token, browse_config)
    suggestions = _call_ebay(suggest_categories, browse_config, app_token, preview.category)

    # File is kept (not deleted) so the confirm step below can reference the image —
    # it's only cleaned up once a listing is actually published, or if this request
    # itself fails (see the except block above).
    return {
        "status": "preview",
        "upload_id": path.stem,
        "item_name": preview.item_name,
        "category_suggestions": [s.model_dump() for s in suggestions],
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


@app.post("/api/identify/confirm")
def confirm_item(payload: ConfirmItemRequest) -> dict:
    """Phase 2 of 2: the full Claude analysis (brand/model/condition/description),
    run only after the user has confirmed a real item name + eBay category — those are
    passed in as ground truth context rather than re-guessed."""
    matches = list(UPLOAD_DIR.glob(f"{payload.upload_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Upload not found or already processed")
    path = matches[0]

    try:
        with _inference_lock:
            local_result = _preprocessor.classify(path)
            gc.collect()
        identification = _call_claude(
            _vision_subagent.identify, path, local_result, payload.item_name, payload.category_name
        )
    except Exception:
        path.unlink(missing_ok=True)
        raise

    identification.category_id = payload.category_id
    return {"status": "complete", "upload_id": path.stem, "result": identification.model_dump()}


def _call_ebay(fn: Callable[..., T], *args, **kwargs) -> T:
    """Run an eBay-backed call, translating failures into clean HTTP errors."""
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        # load_ebay_config()/load_ebay_browse_config()/get_valid_access_token() raise
        # this for missing env vars or a not-yet-connected eBay account.
        raise HTTPException(status_code=400, detail=str(e))
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


class InventoryLocationRequest(BaseModel):
    city: str
    state: str
    postal_code: str
    country: str = "US"
    name: str = "Main Location"
    address_line1: str | None = None
    location_instructions: str | None = None


@app.post("/api/ebay/location")
def create_inventory_location_route(payload: InventoryLocationRequest) -> dict:
    """One-time account setup: registers a single merchant inventory location, which
    eBay's Inventory API requires on every offer before it can be published (separate
    from — and not automatically populated by — any 'ship-from' address configured
    elsewhere in Seller Hub)."""
    config = _call_ebay(load_ebay_config)
    token = _call_ebay(get_valid_access_token, config)
    address = {
        "city": payload.city,
        "stateOrProvince": payload.state,
        "postalCode": payload.postal_code,
        "country": payload.country,
    }
    if payload.address_line1:
        address["addressLine1"] = payload.address_line1
    _call_ebay(
        create_inventory_location, config, token, address, payload.name, payload.location_instructions
    )
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
    price: float
    weight_lbs: float
    currency: str = "USD"


def _public_base_url(request: Request) -> str:
    override = os.environ.get("PUBLIC_BASE_URL")
    return override.rstrip("/") if override else str(request.base_url).rstrip("/")


@app.post("/api/listing/draft")
def create_draft_listing_route(payload: DraftListingRequest, request: Request) -> dict:
    matches = list(UPLOAD_DIR.glob(f"{payload.upload_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Upload not found — it may have already been used or the server restarted")
    path = matches[0]

    image_url = f"{_public_base_url(request)}/uploads/{path.name}"
    result = _call_ebay(
        create_draft_listing,
        payload.identification,
        payload.upload_id,
        image_url,
        payload.price,
        payload.weight_lbs,
        payload.currency,
    )
    # The file is deliberately KEPT here (not deleted) — eBay fetches imageUrls lazily,
    # not synchronously during createOrReplaceInventoryItem, so deleting it this early
    # was causing published listings to show "image not available": by the time eBay
    # actually fetched it (around publish time), the file was already gone. It now only
    # gets deleted once publish_offer_route() actually succeeds (see below), or the
    # identify/refine call that produced it failed outright.
    return {"status": "complete", "result": result.model_dump()}


class UpdateDraftListingRequest(BaseModel):
    identification: ProductIdentification
    upload_id: str
    sku: str
    price: float
    weight_lbs: float
    currency: str = "USD"
    category_query: str | None = None


@app.post("/api/listing/draft/{offer_id}/update")
def update_draft_listing_route(offer_id: str, payload: UpdateDraftListingRequest, request: Request) -> dict:
    """Applies edits made in the review screen (title/brand/condition/description/
    category/price/weight) to the already-created draft, in place — the automatic
    identification and category suggestion can be wrong, and this is the human's chance
    to fix it before publishing."""
    matches = list(UPLOAD_DIR.glob(f"{payload.upload_id}.*"))
    if not matches:
        raise HTTPException(
            status_code=404,
            detail="Upload not found — the image may have been removed. Start a new draft instead.",
        )
    path = matches[0]

    image_url = f"{_public_base_url(request)}/uploads/{path.name}"
    result = _call_ebay(
        update_draft_listing,
        offer_id,
        payload.sku,
        payload.identification,
        image_url,
        payload.price,
        payload.weight_lbs,
        payload.currency,
        1,
        payload.category_query,
    )
    return {"status": "complete", "result": result.model_dump()}


@app.get("/api/listing/draft/{offer_id}")
def get_draft_listing_route(offer_id: str) -> dict:
    """Fetch a created draft straight from eBay's API — a reliable way to confirm
    it exists without depending on eBay's sandbox Seller Hub web UI, which is known
    to be far less complete/reliable than production's."""
    config = _call_ebay(load_ebay_config)
    token = _call_ebay(get_valid_access_token, config)
    offer = _call_ebay(get_offer, config, token, offer_id)
    return {"status": "complete", "result": offer}


class PublishRequest(BaseModel):
    upload_id: str | None = None


@app.post("/api/listing/publish/{offer_id}")
def publish_offer_route(offer_id: str, payload: PublishRequest) -> dict:
    """Make a previously-created draft offer live and publicly purchasable on
    eBay. eBay's Seller Hub has no view for API-created unpublished offers, so
    this app's own listing-preview screen is the Human-in-the-Loop review step
    for this action — this route is only ever called from an explicit user
    click after that review, never automatically."""
    config = _call_ebay(load_ebay_config)
    token = _call_ebay(get_valid_access_token, config)
    result = _call_ebay(publish_offer, config, token, offer_id)
    # Only now — once eBay has actually taken the listing live — is it safe to remove
    # the source image. Deleting it any earlier (as create_draft_listing_route used to)
    # meant eBay's lazy imageUrls fetch could happen after the file was already gone.
    if payload.upload_id:
        for match in UPLOAD_DIR.glob(f"{payload.upload_id}.*"):
            match.unlink(missing_ok=True)
    return {"status": "complete", "result": result.model_dump()}
