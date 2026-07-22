# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AgentX (repo name `EbayAiX`) is a portfolio project demonstrating production-grade AI agent architecture: an
e-commerce helper that ingests a photo of a physical item, identifies it, prices it, and drafts a live eBay
listing. It's built to showcase Agent Architecture Design, Computer Vision (local + cloud), Agentic Search,
and AI Governance (Human-in-the-Loop controls) as resume-relevant skills, so code should reflect deliberate,
enterprise-style patterns rather than the shortest path to a working demo.

So far: local image classification, Claude-based structured identification, a low-cost eBay comparable-
listings lookup (no LLM involved), a FastAPI front end chaining all of it with a human clarification step,
and an eBay OAuth 2.0 connect flow. The orchestrator, actual eBay listing-creation calls, and the HITL
approval gate in front of them (described below under "Planned architecture") are not yet implemented —
eBay's Sell APIs themselves haven't been called yet, only the auth handshake.

**History note:** pricing was originally a Claude-based subagent doing agentic web search/fetch against
eBay's sold-listings pages. It got expensive fast — an uncapped `web_fetch` burned through the entire
Anthropic account balance in a couple of test runs, because a real eBay search-results page is huge and
server-side tool results stay in context for every subsequent step within the same request. It was replaced
outright with the plain `src/ebay/browse.py` approach below per explicit direction: no LLM call for pricing
at all, just eBay's own Browse API showing a few comparable listings and their asking prices.

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
`.env` for the OAuth connect flow to work. `EBAY_RU_NAME` is not a URL — it's the RuName string eBay
generates for a registered redirect URI. Separately, `EBAY_PROD_APP_ID`/`EBAY_PROD_CERT_ID` (your
**production** keyset, not sandbox) must be set for `src/ebay/browse.py`'s comp search — see .env.example
and the Architecture section below for why this specifically needs production credentials.

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

**`src/ebay/browse.py`** — no Claude involved at all; a plain, cheap eBay Browse API call.
`search_comparable_listings(query)` gets an application-level access token via the **Client Credentials**
grant (`get_application_access_token()` — a different, simpler auth flow than the user OAuth handshake
below: no browser consent step, just a server-to-server token exchange, cached in memory until near
expiry), then calls `GET /buy/browse/v1/item_summary/search` and returns up to 3 `EbayComp` (title, price,
currency, condition, item URL). `src/web/app.py`'s `POST /api/price` builds the search query from the
`ProductIdentification` (`{brand} {model_number}` when both are present, falling back to item name) and
returns the comps directly — no LLM call, no token cost beyond the eBay API request itself.

**Always uses PRODUCTION eBay credentials** (`EBAY_PROD_APP_ID`/`EBAY_PROD_CERT_ID`), regardless of the rest
of the app's `EBAY_ENVIRONMENT` setting. eBay's sandbox environment has essentially no real search/catalog
data — `item_summary/search` reliably returns `total: 0` there, a well-documented, longstanding eBay
limitation (confirmed against multiple independent developer reports), not a bug in this code. This is a
read-only, public-data search with no side effects, so production credentials here carry none of the risk
that using production for listing creation would.

**`src/web/app.py`** — the FastAPI front end. `POST /api/identify` saves the upload to `data/uploads/`
(git-ignored, size-capped at 10MB, extension-allowlisted to jpg/jpeg/png/webp), runs the local classifier +
Vision Subagent, and returns the result directly if `identification_confidence` is `"medium"`/`"high"`. If
it's `"low"`, it instead returns `status: "needs_clarification"` plus an `upload_id` and leaves the file on
disk. The frontend (`src/web/static/index.html`, vanilla JS) then prompts the user for the item's name and
posts it to `POST /api/identify/refine`, which re-runs `VisionSubagent.identify(..., user_provided_name=...)`
with that hint folded into the prompt and always returns a final result (no further clarification loop).
Both endpoints delete the uploaded file once they return a final result. Once identification is final, the
frontend shows a "Show Comparable Listings" button that posts the `ProductIdentification` (FastAPI validates
the JSON body directly against that Pydantic model) to `POST /api/price`, which calls
`search_comparable_listings()` above. The `VisionPreprocessor`/`VisionSubagent` instances are constructed
once at module import time and reused across requests — re-instantiating per request would reload the
ResNet50 weights every call. `_call_claude()` wraps every Claude-backed call (identify/refine — pricing no
longer uses Claude) and translates `OverloadedError`/`RateLimitError`/`APIConnectionError`/`APIStatusError`
into a clean `503`/`502` HTTP response instead of a bare 500 — this was added after a real Anthropic-side
outage surfaced as an unhandled exception during development.

**`src/ebay/`** — two independent pieces. **User OAuth 2.0** (authorization-code grant, no listing calls
yet): `config.py`'s `EbayConfig`/`load_ebay_config()` resolves sandbox vs. production base/token/authorize
URLs; `oauth.py` builds the consent URL and does the `authorization_code`/`refresh_token` exchanges against
eBay's Identity API (`EbayTokens` tracks both tokens' expiry; eBay doesn't rotate the refresh token on
refresh); `token_store.py` persists tokens to git-ignored `data/ebay_tokens.json` (single-user, local/small
deployment — not a multi-tenant design) and exposes `get_valid_access_token()`, which transparently
refreshes an expired access token before returning it; `src/web/ebay_routes.py` wires this into FastAPI:
`GET /ebay/connect` (redirect to eBay, with CSRF `state`), `GET /ebay/callback` (code -> tokens), `GET
/ebay/status`. **Application-level Browse API access** (separate, see `browse.py` above): `config.py`'s
`EbayBrowseConfig`/`load_ebay_browse_config()`, always production, no user consent involved.

**Package layout is deliberate:** `src/`, `src/ml/`, `src/agents/`, `src/web/`, `src/ebay/` have no
`__init__.py` — they work as Python 3.14 implicit namespace packages. Do not add empty `__init__.py` files
back in; only add one if it needs to hold real code.

### Planned architecture (not yet built)

Per the original design: a hierarchical Orchestrator-Workers pattern where a Claude-based orchestrator
routes between the Vision Subagent (above) and eBay Sell REST API calls (using the OAuth tokens `src/ebay/`
already handles) to create draft listings — gated behind a mandatory Human-in-the-Loop manual approval step
before any live write to eBay. Pricing (`browse.py`) is intentionally plain eBay API + Python, not an
orchestrated subagent — no LLM reasoning involved.
