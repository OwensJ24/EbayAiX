"""FastAPI app: image upload -> local classifier -> Claude Vision Subagent -> Pricing Subagent."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable, TypeVar

import anthropic
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.pricing_subagent import PricingSubagent
from src.agents.vision_subagent import ProductIdentification, VisionSubagent
from src.ml.vision_preprocessor import ClassificationResult, VisionPreprocessor
from src.web.ebay_routes import router as ebay_router

T = TypeVar("T")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = _PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AgentX")
app.include_router(ebay_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Loaded once at process startup — ResNet50 weight loading and API client
# construction are both too expensive to redo per request.
_preprocessor = VisionPreprocessor()
_vision_subagent = VisionSubagent()
_pricing_subagent = PricingSubagent()


class RefineRequest(BaseModel):
    upload_id: str
    item_name: str


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


def _save_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'unknown'}")

    contents = _read_upload_within_limit(file, MAX_UPLOAD_BYTES)
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
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e.message}")


@app.post("/api/identify")
def identify(file: UploadFile = File(...)) -> dict:
    path = _save_upload(file)
    try:
        local_result = _preprocessor.classify(path)
        identification = _call_claude(_vision_subagent.identify, path, local_result)
    except Exception:
        path.unlink(missing_ok=True)
        raise

    if identification.identification_confidence == "low":
        return {
            "status": "needs_clarification",
            "upload_id": path.stem,
            "result": identification.model_dump(),
        }

    path.unlink(missing_ok=True)
    return {"status": "complete", "result": identification.model_dump()}


@app.post("/api/identify/refine")
def refine(payload: RefineRequest) -> dict:
    matches = list(UPLOAD_DIR.glob(f"{payload.upload_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Upload not found or already processed")
    path = matches[0]

    try:
        local_result = _preprocessor.classify(path)
        identification = _call_claude(
            _vision_subagent.identify, path, local_result, user_provided_name=payload.item_name
        )
    finally:
        path.unlink(missing_ok=True)

    return {"status": "complete", "result": identification.model_dump()}


@app.post("/api/price")
def price(identification: ProductIdentification) -> dict:
    pricing = _call_claude(_pricing_subagent.research_price, identification)
    return {"status": "complete", "result": pricing.model_dump()}
