# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AgentX (repo name `EbayAiX`) is a portfolio project demonstrating production-grade AI agent architecture: an
e-commerce helper that ingests a photo of a physical item, identifies it, prices it, and drafts a live eBay
listing. It's built to showcase Agent Architecture Design, Computer Vision (local + cloud), Agentic Search,
and AI Governance (Human-in-the-Loop controls) as resume-relevant skills, so code should reflect deliberate,
enterprise-style patterns rather than the shortest path to a working demo.

So far: local image classification, Claude-based structured identification, and a FastAPI upload front end
that chains the two with a human clarification step. The pricing/research subagent, orchestrator, eBay Sell
API integration, and eBay-listing HITL approval gate described below under "Planned architecture" are not
yet implemented.

## Commands

Environment is managed by `uv` (Python 3.14, `.venv/`). No test suite or linter is configured yet.

```bash
uv sync                                   # install/sync dependencies from pyproject.toml / uv.lock
uv add <package>                          # add a new dependency

# Local ResNet50 classifier, standalone
uv run python -m src.ml.vision_preprocessor <image_path> [--top-k N] [--device cpu|mps]

# Full pipeline: local classifier -> Claude Vision Subagent
uv run python -m src.agents.vision_subagent <image_path> [--model claude-sonnet-5]

# Web app (upload UI + API), served at http://127.0.0.1:8000
uv run uvicorn src.web.app:app --reload
```

`ANTHROPIC_API_KEY` must be set in `.env` (see `.env.example`) for anything in `src/agents/` to work —
loaded via `python-dotenv`.

**macOS SSL gotcha:** a python.org framework build of Python has no CA bundle configured by default, so the
first `torch.hub` weights download in `vision_preprocessor.py` can fail with `SSLCertVerificationError`. Fix
by pointing at the `certifi` bundle already pulled in as an `anthropic` dependency:
```bash
SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())") uv run python -m src.ml.vision_preprocessor <image_path>
```
Only needed once — the ResNet50 weights are then cached under `~/.cache/torch/hub/checkpoints/`.

## Architecture

Two independent stages exist today, chained by `src/agents/vision_subagent.py`'s `main()`:

**`src/ml/vision_preprocessor.py`** — local-only, no network calls. `VisionPreprocessor` loads a pretrained
ResNet50 (`torchvision.models.ResNet50_Weights.DEFAULT`) once, auto-selects `mps`/`cpu`, and exposes
`classify(image_path) -> ClassificationResult`. `ClassificationResult`/`Prediction` are dataclasses with
`to_dict()`/`to_json()` — this is the JSON-serializable "local ML metadata" handed to the Claude layer.

**`src/agents/vision_subagent.py`** — calls the Claude API. `VisionSubagent.identify(image_path,
local_classification)` base64-encodes the image, sends it to Claude alongside the local classifier's JSON
output as context, and uses structured outputs (`client.messages.parse(..., output_format=ProductIdentification)`)
to force a strict schema: item name, brand, model number, category, a constrained condition enum, condition
notes, distinguishing features, and a confidence level. The system prompt explicitly tells Claude to trust
its own visual read over the local classifier's guess when they disagree, and to return `null` for
brand/model number rather than guess.

**`src/web/app.py`** — the FastAPI front end. `POST /api/identify` saves the upload to `data/uploads/`
(git-ignored, size-capped at 10MB, extension-allowlisted to jpg/jpeg/png/webp), runs the same two stages
above, and returns the result directly if `identification_confidence` is `"medium"`/`"high"`. If it's
`"low"`, it instead returns `status: "needs_clarification"` plus an `upload_id` and leaves the file on disk.
The frontend (`src/web/static/index.html`, vanilla JS) then prompts the user for the item's name and posts
it to `POST /api/identify/refine`, which re-runs `VisionSubagent.identify(..., user_provided_name=...)` with
that hint folded into the prompt and always returns a final result (no further clarification loop). Both
endpoints delete the uploaded file once they return a final result. The `VisionPreprocessor`/`VisionSubagent`
instances are constructed once at module import time and reused across requests — re-instantiating per
request would reload the ResNet50 weights every call.

**Package layout is deliberate:** `src/`, `src/ml/`, `src/agents/`, `src/web/` have no `__init__.py` — they
work as Python 3.14 implicit namespace packages. Do not add empty `__init__.py` files back in; only add one
if it needs to hold real code.

### Planned architecture (not yet built)

Per the original design: a hierarchical Orchestrator-Workers pattern where a Claude-based orchestrator
routes between the Vision Subagent (above), a Pricing/Research Subagent (agentic web search for historical
eBay sold-listing data), and eBay Sell REST API integration (OAuth 2.0 user tokens) to create draft
listings — gated behind a mandatory Human-in-the-Loop manual approval step before any live write to eBay.
