# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AgentX (repo name `EbayAiX`) is a portfolio project demonstrating production-grade AI agent architecture: an
e-commerce helper that ingests a photo of a physical item, identifies it, prices it, and drafts a live eBay
listing. It's built to showcase Agent Architecture Design, Computer Vision (local + cloud), Agentic Search,
and AI Governance (Human-in-the-Loop controls) as resume-relevant skills, so code should reflect deliberate,
enterprise-style patterns rather than the shortest path to a working demo.

So far: local image classification, Claude-based structured identification, a Pricing/Research Subagent
(agentic web search for resale comps), a FastAPI front end chaining all three with a human clarification
step, and an eBay OAuth 2.0 connect flow. The orchestrator, actual eBay listing-creation calls, and the HITL
approval gate in front of them (described below under "Planned architecture") are not yet implemented —
eBay's Sell APIs themselves haven't been called yet, only the auth handshake.

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

`EBAY_APP_ID`, `EBAY_CERT_ID`, `EBAY_RU_NAME` (and `EBAY_ENVIRONMENT`, default `sandbox`) must be set in
`.env` for anything in `src/ebay/` to work. `EBAY_RU_NAME` is not a URL — it's the RuName string eBay
generates for a registered redirect URI. See `.env.example`.

## Deployment

`Dockerfile` builds the whole app (FastAPI + PyTorch) as one image, targeting Render (see `render.yaml`) —
chosen over serverless platforms (Vercel, etc.) because PyTorch/ResNet50 and the file-based upload/token
storage don't fit a stateless serverless model. Key details baked into the Dockerfile:

- **CPU-only PyTorch on Linux.** `pyproject.toml`'s `[tool.uv.sources]` routes `torch`/`torchvision` to
  `https://download.pytorch.org/whl/cpu` when `platform_system != 'Darwin'`. Without this, `uv` resolves the
  default CUDA build on Linux — several GB of unneeded `nvidia-*` packages. macOS dev is unaffected (no CUDA
  variant exists there anyway; MPS still works locally).
- **ResNet50 weights are baked in at build time** (`RUN uv run python -c "from src.ml.vision_preprocessor
  import VisionPreprocessor; VisionPreprocessor()"`) so a cold start on Render's free tier (which spins down
  after inactivity) doesn't need a ~100MB download before the server can respond.
- Test locally with `docker build -t agentx-test .` then `docker run -p 8001:8000 --env-file .env agentx-test`.

**eBay's OAuth redirect requires real HTTPS** — it will not redirect to a `localhost` URL no matter how
the RuName's "Auth accepted URL" is configured (eBay silently shows its own generic confirmation page
instead). This is one of the reasons the app needs a real HTTPS deployment rather than only running locally.

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

**`src/agents/pricing_subagent.py`** — calls the Claude API, sourcing pricing from eBay only (never general
web results). `PricingSubagent.research_price(identification)` builds eBay's own sold-listings search URLs
(`ebay.com/sch/i.html?_nkw=<query>&LH_Sold=1&LH_Complete=1`) from the identification in Python and embeds
them directly in the prompt — `web_fetch` can only retrieve URLs already present in the conversation, so
these are constructed ourselves rather than left for Claude to guess at. Claude fetches those (via
`web_fetch_20260209`, `allowed_domains: ["ebay.com"]`) and reads the actual sold listings; `web_search_20260209`
(same domain restriction) is available as a fallback if the direct URLs come up empty. Each comparable
listing carries an explicit `listing_type: "sold" | "active"` so the sold-vs-active distinction is a
structured field, not just implied by prose reasoning — the model is instructed to fall back to active
listings only when it truly can't find sold comps, and to cap `pricing_confidence` accordingly when it does.

Structured outputs (`output_format=`) aren't used here — tool use plus `output_config.format` in the same
request is a combination worth double-checking against the current API docs before relying on it — so this
instead prompts for raw JSON and validates it via `PricingRecommendation.model_validate()` after a small
`_extract_json` cleanup step (strips markdown code fences models sometimes add despite being told not to).
**Take the LAST text block from `response.content`, not the first.** With multiple tool-call rounds (fetch,
possibly search too), Claude often emits an early text block like "Let me check eBay's sold listings..."
before any tool use; grabbing the first text block gets that commentary instead of the final JSON-only
answer, and fails with a JSON parse error that looks identical to a genuinely empty response. This was a
real bug caught via a production traceback, not a hypothetical.

**`src/web/app.py`** — the FastAPI front end. `POST /api/identify` saves the upload to `data/uploads/`
(git-ignored, size-capped at 10MB, extension-allowlisted to jpg/jpeg/png/webp), runs the local classifier +
Vision Subagent, and returns the result directly if `identification_confidence` is `"medium"`/`"high"`. If
it's `"low"`, it instead returns `status: "needs_clarification"` plus an `upload_id` and leaves the file on
disk. The frontend (`src/web/static/index.html`, vanilla JS) then prompts the user for the item's name and
posts it to `POST /api/identify/refine`, which re-runs `VisionSubagent.identify(..., user_provided_name=...)`
with that hint folded into the prompt and always returns a final result (no further clarification loop).
Both endpoints delete the uploaded file once they return a final result. Once identification is final, the
frontend shows a "Get Price Estimate" button that posts the `ProductIdentification` (FastAPI validates the
JSON body directly against that Pydantic model) to `POST /api/price`, which runs the Pricing Subagent. The
`VisionPreprocessor`/`VisionSubagent`/`PricingSubagent` instances are all constructed once at module import
time and reused across requests — re-instantiating per request would reload the ResNet50 weights every
call. `_call_claude()` wraps every subagent call and translates `OverloadedError`/`RateLimitError`/
`APIConnectionError`/`APIStatusError` into a clean `503`/`502` HTTP response instead of a bare 500 — this
was added after a real Anthropic-side outage surfaced as an unhandled exception during development.

**`src/ebay/`** — eBay OAuth 2.0 (authorization-code grant for user access tokens), no listing calls yet.
`config.py` loads `EbayConfig` from env and resolves sandbox vs. production base/token/authorize URLs.
`oauth.py` builds the consent URL and does the `authorization_code`/`refresh_token` exchanges against
eBay's Identity API (`EbayTokens` tracks both tokens' expiry; eBay doesn't rotate the refresh token on
refresh). `token_store.py` persists tokens to git-ignored `data/ebay_tokens.json` (single-user, local/small
deployment — not a multi-tenant design) and exposes `get_valid_access_token()`, which transparently
refreshes an expired access token before returning it. `src/web/ebay_routes.py` wires this into FastAPI:
`GET /ebay/connect` (redirect to eBay, with CSRF `state`), `GET /ebay/callback` (code -> tokens), `GET
/ebay/status`.

**Package layout is deliberate:** `src/`, `src/ml/`, `src/agents/`, `src/web/`, `src/ebay/` have no
`__init__.py` — they work as Python 3.14 implicit namespace packages. Do not add empty `__init__.py` files
back in; only add one if it needs to hold real code.

### Planned architecture (not yet built)

Per the original design: a hierarchical Orchestrator-Workers pattern where a Claude-based orchestrator
routes between the Vision Subagent, the Pricing Subagent (both above), and eBay Sell REST API calls (using
the OAuth tokens `src/ebay/` already handles) to create draft listings — gated behind a mandatory
Human-in-the-Loop manual approval step before any live write to eBay.
