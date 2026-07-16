"""FastAPI app: image upload -> local classifier -> Claude Vision Subagent."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.vision_subagent import VisionSubagent
from src.ml.vision_preprocessor import VisionPreprocessor

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = _PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AgentX")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Loaded once at process startup — ResNet50 weight loading and API client
# construction are both too expensive to redo per request.
_preprocessor = VisionPreprocessor()
_vision_subagent = VisionSubagent()


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


@app.post("/api/identify")
def identify(file: UploadFile = File(...)) -> dict:
    path = _save_upload(file)
    local_result = _preprocessor.classify(path)
    identification = _vision_subagent.identify(path, local_result)

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

    local_result = _preprocessor.classify(path)
    identification = _vision_subagent.identify(path, local_result, user_provided_name=payload.item_name)

    path.unlink(missing_ok=True)
    return {"status": "complete", "result": identification.model_dump()}
