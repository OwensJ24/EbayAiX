# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AgentX (repo name `EbayAiX`) is a portfolio project demonstrating production-grade AI agent architecture: an
e-commerce helper that ingests a photo of a physical item, identifies it, prices it, and drafts a live eBay
listing. It's built to showcase Agent Architecture Design, Computer Vision (local + cloud), Agentic Search,
and AI Governance (Human-in-the-Loop controls) as resume-relevant skills, so code should reflect deliberate,
enterprise-style patterns rather than the shortest path to a working demo.

So far: local image classification, a two-phase Claude-based structured identification (a cheap name+category
guess, confirmed/corrected by a human, then a fuller analysis informed by that confirmation), a low-cost eBay
comparable-listings lookup (no LLM involved), eBay draft-listing creation (Inventory Item + Offer), an in-app
listing preview + explicit publish step, a FastAPI front end chaining all of it together, and an eBay OAuth
2.0 connect flow. The Claude-based orchestrator tying these steps together programmatically, and a more
general HITL approval gate, are still future work (see "Planned architecture") — but both the mandatory
identify/confirm step and the in-app review-then-publish flow already built are this project's concrete
instances of that gate.

**Important finding that shaped the publish flow:** eBay's Seller Hub has no UI at all for reviewing an
unpublished offer created via the Inventory API — confirmed via multiple independent eBay developer
community threads, not a bug or misconfiguration on this project's side. An earlier version of this project
linked to a Seller Hub "drafts" URL after creating a draft; that link was based on an incorrect assumption
and has been removed. Because eBay provides no native review surface for API-created offers, this app's own
frontend is the Human-in-the-Loop review point: it shows a full preview of everything that will go into the
listing, and only an explicit, checkbox-gated user action calls `publishOffer` (see `src/ebay/listing.py`
below) to make it live.

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
and the Architecture section below for why this specifically needs production credentials. `PUBLIC_BASE_URL`
is optional — set it explicitly if the app's own request-derived base URL is ever wrong for building the
publicly-fetchable image URLs eBay needs when creating a draft listing (see the HTTPS scheme note below for
why this was silently broken until fixed at the `Dockerfile` level).

**If a previously-connected eBay account stops seeing business policies/location/category suggestions in
draft listings**, it's because the OAuth scope was widened after that connection was made
(`sell.account.readonly`, then later the base `https://api.ebay.com/oauth/api_scope` needed for the Taxonomy
API's category suggestions — see `config.py`'s `DEFAULT_SCOPES`) — the stored refresh token doesn't
retroactively gain scopes. Fix: click "Connect eBay account" again.

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
- **uvicorn is started with `--proxy-headers --forwarded-allow-ips='*'`.** Render terminates TLS at its edge
  and forwards plain HTTP internally, so without trusting `X-Forwarded-Proto`, `request.base_url` (used by
  `_public_base_url()` in `app.py` to build the `imageUrls` eBay fetches) silently resolves to `http://`
  instead of `https://` — uvicorn's default trusted-proxy list is only `127.0.0.1`, which doesn't match
  Render's actual internal proxy IP. This was a real, confirmed bug: eBay's Inventory API requires HTTPS
  image URLs and silently drops non-HTTPS ones (no error — the listing just publishes with a broken image),
  which is exactly what was happening. `--forwarded-allow-ips='*'` is safe specifically because this
  container has no other direct public exposure — the only thing that can reach it is Render's own edge.
  Confirmed via local repro (`curl -H "X-Forwarded-Proto: https"` against the built image) both before and
  after adding the flag. If image issues ever recur, setting `PUBLIC_BASE_URL` explicitly sidesteps this
  proxy-trust question entirely, at the cost of one more env var to keep in sync with the deployed domain.
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

**`src/agents/vision_subagent.py`** — calls the Claude API, in two deliberate phases with a mandatory human
checkpoint between them, per explicit direction: name and category affect everything downstream (comparable-
listings search, eBay category-specific condition/aspect rules, buyer discoverability), so they're confirmed
*first* rather than being just another guess the user might fix later.

`VisionSubagent.preview(image_path, local_classification) -> ItemPreview` — a cheap first pass
(`max_tokens=256`) that returns only `item_name` + a free-text `category` guess, base64-encoding the image
and using the local classifier's JSON output as context exactly like `identify()` below. This is what
`POST /api/identify` calls; the result is never used directly for listing creation, only to seed the
confirm-step UI (see `app.py` below).

`VisionSubagent.identify(image_path, local_classification, confirmed_item_name, confirmed_category) ->
ProductIdentification` — the fuller analysis, only ever run *after* a human has confirmed/edited the name and
picked a real eBay category (see `app.py`'s `/api/identify/confirm`). `confirmed_item_name`/`confirmed_category`
are required params, not an optional override — the system prompt tells Claude to treat them as ground truth
and analyze everything else (brand, model number, condition, description); after parsing, `identify()`
defensively overwrites `item_name`/`category` on the result with the confirmed strings rather than trusting
the model to echo them back exactly. Uses structured outputs
(`client.messages.parse(..., output_format=ProductIdentification)`) to force a strict schema: item name,
brand, model number, category, `category_id` (the confirmed real eBay categoryId, set by the caller —
Claude never produces this), a constrained condition enum, condition notes, a buyer-facing
`content_description`, distinguishing features, and a confidence level. Be conservative about condition:
only claim 'New' if there is clear evidence (tags, packaging, no wear); return `null` for brand/model number
rather than guess.

**`content_description` vs. `condition_notes` — deliberately separate, per explicit direction.**
`content_description` is what actually becomes the eBay listing description (see `_build_description()` in
`listing.py` below) — it must describe what the item is and what's visible/included (color, accessories,
ports, packaging contents), and the system prompt explicitly tells Claude not to mention wear, damage, or
condition there. `condition_notes` stays as condition-rating justification only (and still feeds
`resolve_aspects()`'s text-matching corpus in `listing.py`) — it's no longer blended into the buyer-facing
description the way it used to be.

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
(git-ignored, size-capped at 10MB, extension-allowlisted to jpg/jpeg/png/webp — all formats eBay's Inventory
API accepts, confirmed against eBay's own docs) after `_validate_image_dimensions()` opens it with Pillow and
rejects anything under `EBAY_MIN_IMAGE_DIMENSION` (500px) on either side with a clear 400 — eBay's own docs
say a smaller image "may be blocked" from the listing, silently, which is a worse failure mode (a seemingly
successful publish with a broken image) than rejecting it up front at upload time. Then runs the local
classifier + `VisionSubagent.preview()` (Phase 1 — cheap name+category guess, see `vision_subagent.py`
above), followed immediately by `suggest_categories()` (`listing.py`, using the *application-level*
`browse.py` token, not the user's OAuth one — Taxonomy category suggestions are public data, so this works
even before the user has connected their own eBay account, same reasoning as `browse.py`'s comp pricing) to
turn that free-text category guess into a handful of real, valid eBay categories. Returns
`{status: "preview", upload_id, item_name, category_suggestions}` — always this shape, no
confidence-based branching anymore (the mandatory confirm step below replaces what used to be a
conditional "needs_clarification" path for low-confidence guesses only).

**New `POST /api/categories/search`** — same `suggest_categories()` call, for the frontend's re-search box
when none of the initial suggestions fit different search terms than the LLM's own guess.

**`POST /api/identify/confirm`** (this used to be `/api/identify/refine`, with a materially different
contract — renamed since it's no longer "refine a low-confidence guess" but "run the full analysis now that
name+category are confirmed"). Body: `{upload_id, item_name, category_id, category_name}` — all mandatory,
never optional. Re-runs the local classifier, then calls `VisionSubagent.identify(path, local_result,
item_name, category_name)` (Phase 2 — the full analysis, informed by the *confirmed* name/category rather
than re-guessing them), sets `identification.category_id = category_id` on the result (Claude never produces
this — it's the real eBay Taxonomy ID the user picked), and returns the final `ProductIdentification`. Every
downstream step (`/api/price`, `/api/listing/draft`, the edit-before-publish screen,
`/api/listing/publish/{offer_id}`) is unchanged — they already just consume whatever `ProductIdentification`
is in the frontend's `currentIdentification`.

**Both `/api/identify` and `/api/identify/confirm` keep the uploaded file on disk** (returning `upload_id` on
every response) — it stays on disk through the confirm step, draft creation, *and* any edits, and is only
deleted once `publish_offer_route()` actually succeeds, or the identify/confirm call itself fails. It used to
be deleted right after draft creation, which was a real bug: eBay fetches `imageUrls` lazily rather than
synchronously during `createOrReplaceInventoryItem`, so by the time eBay actually fetched the image (around
publish time), the file was already gone — surfacing as "image not available" on real, live, published
listings. A user who abandons the flow entirely (never publishes) leaves an orphaned file in
`data/uploads/`; no cleanup job exists for this (accepted tradeoff, not a bug).

**Local ResNet50 inference is serialized across requests via `_inference_lock` (a plain `threading.Lock`).**
Each sync FastAPI route runs in its own thread-pool thread, so without this, multiple images uploaded in
quick succession would run CPU inference concurrently — observed causing real OOM crashes on Render's
free-tier memory limit, since each PyTorch CPU inference pass is memory-hungry enough that 2-3 running at
once exceeds it. The lock only wraps `_preprocessor.classify()`, not the (network-bound) Claude call, so
those still run concurrently. `VisionPreprocessor.__init__` also sets `torch.set_num_threads(1)` on CPU to
cut per-inference thread/memory overhead further. The frontend adds a client-side guard on top (disables the
dropzone/confirm button while a request is in flight) so this is defense-in-depth, not the only line of
defense — a different client hitting the API directly still can't cause concurrent local inference.

**A second, distinct OOM pattern: crashes reliably on the *second* full identify pass, even with zero
concurrency** (e.g. completing one flow, refreshing the page, then starting another one). This is a known
glibc/PyTorch behavior on constrained containers, not a code bug in the traditional sense — glibc's malloc
creates multiple memory arenas per process, and memory freed within a non-primary arena often isn't returned
to the OS, so RSS ratchets upward across requests instead of resetting after each one, until it exceeds
Render's free-tier 512MB limit. Two mitigations: `Dockerfile` sets `ENV MALLOC_ARENA_MAX=1` (caps glibc to a
single arena it can actually release memory from — verified via `docker run` that the env var reaches the
container); `app.py`'s `identify()`/`refine()` call `gc.collect()` immediately after `_preprocessor.classify()`
(still inside `_inference_lock`, so it can't race a concurrent classify() call), forcing CPython to release
the heaviest per-request allocation (the image tensor + inference pass) promptly rather than waiting on
normal refcounting/GC timing. If OOM crashes persist after this, the free tier's 512MB may simply be
insufficient for this stack (PyTorch + ResNet50 + FastAPI + Anthropic/httpx clients) — upgrading Render's
plan would be the next lever, not further code changes.

Once the confirm step (above) returns a final `ProductIdentification`, the frontend automatically posts it
(FastAPI validates the JSON body directly against that Pydantic model) to
`POST /api/price`, which calls `search_comparable_listings()` above using a query built by
`browse.build_query()` (shared with the Taxonomy category lookup in `listing.py`, see below). The average of
the returned comps pre-fills an editable price field. The user must also enter a package weight (lbs) — a
real, blocking eBay requirement (`packageWeightAndSize.weight`, `errorId 25020` "package weight is not valid
or is missing" if omitted, discovered from a real production error — the Vision Subagent has no way to
estimate this from a photo, so it's the one piece of listing data the human always has to supply directly
rather than something the pipeline can infer). A "Create Draft Listing" button posts
`{identification, upload_id, price, weight_lbs, currency}` to `POST /api/listing/draft`, which calls
`create_draft_listing()` (see `src/ebay/listing.py` below) using the still-on-disk image, served publicly at
`GET /uploads/{filename}` (a `StaticFiles` mount — upload IDs are `uuid4().hex`, unguessable enough that
serving them without auth is an accepted tradeoff for this portfolio project's scale).

Once a draft is created, the frontend shows an **editable** review form (title, brand, condition dropdown,
eBay category search terms, description, plus the existing price/weight inputs), pre-filled from the same
`ProductIdentification` already in memory, plus any `missing`/`notes` gaps and resolved `aspects` from the
draft result. This form, not eBay's Seller Hub, is the human review *and correction* point (see the Seller
Hub finding above) — the automatic identification/category-suggestion pipeline can be wrong, so every field
that matters for a correct listing (title, brand, condition, category, description) is editable here, not
just visible. A "Save Changes" button posts the edited fields to
`POST /api/listing/draft/{offer_id}/update`, which calls `update_draft_listing()` (see `src/ebay/listing.py`
below) — this updates the *same* SKU/offer in place (both underlying eBay calls are idempotent PUTs), so
edits don't create a duplicate listing. The edited description textarea becomes the new `condition_notes`
directly (with `distinguishing_features` cleared) rather than being reconstructed from separate fields, so a
second edit doesn't duplicate content. Publishing requires checking an explicit confirmation checkbox ("I
understand this will create a real, live eBay listing...") before the "Publish to eBay" button enables;
clicking it posts to `POST /api/listing/publish/{offer_id}` (now with `{upload_id}` in the body — see the
image-lifecycle note above), which calls `publish_offer()` (see `src/ebay/listing.py` below) and returns a
real, public `listing_url` on success.

**The confirmation checkbox is disabled outright (not just a warning) whenever `result.missing` is
non-empty.** `category`/`merchant_location`/`listing_policies` are each hard requirements for *publishing* an
offer (unlike creating the draft, which succeeds without them — see `create_draft_listing()` below), so
letting the user attempt to publish anyway just wastes a round-trip to eBay for a guaranteed failure. This
was discovered from a real production error: `publishOffer` returned `errorId 25002` with the message
`"No <Item.Country> exists..."`, which sounds like a missing address field but actually means **no inventory
location (merchantLocationKey) is set up on the eBay account** — confirmed via eBay developer community
threads, the same kind of misleadingly-worded eBay error as the earlier `Content-Language`/25709 case. An
eBay account's general "ship-from" address (set elsewhere in Seller Hub) is a distinct concept from an
Inventory API location and doesn't populate this automatically.

**`POST /api/ebay/location`** (frontend: a collapsed "Set up eBay shipping location" `<details>` block above
the upload dropzone) closes this gap directly rather than sending the user to Seller Hub: it calls
`create_inventory_location()` (`src/ebay/listing.py`) — `POST
/sell/inventory/v1/location/{merchantLocationKey}` with the address the user types in (city/state/postal
code/country required, address line 1 optional) — registering one fixed location,
`DEFAULT_MERCHANT_LOCATION_KEY = "agentx-default-location"`. One fixed key (not per-listing, not
user-chosen) is deliberate: this is a single-seller portfolio app, so one location is all it ever needs.
This is meant to be run once per eBay account; `get_merchant_location_key()` then finds it automatically on
every subsequent draft via the existing `GET /sell/inventory/v1/location` lookup.

The `VisionPreprocessor`/`VisionSubagent` instances are constructed once at module import time and reused
across requests — re-instantiating per request would reload the ResNet50 weights every call. `_call_claude()`
wraps every Claude-backed call and `_call_ebay()` wraps every eBay-backed call, each translating their
respective API failures into clean `4xx`/`502`/`503` HTTP responses instead of a bare 500 —
`_call_claude()` was added after a real Anthropic-side outage surfaced as an unhandled exception during
development; `_call_ebay()` additionally catches the `RuntimeError` that `load_ebay_config()`/
`load_ebay_browse_config()`/`get_valid_access_token()` raise for missing env vars or a not-yet-connected
eBay account — this used to reach the client as a bare 500 before `_call_ebay()` existed.

**`src/ebay/listing.py`** — creates a draft eBay listing, lets it be edited in place, and separately,
optionally publishes it. The category/condition/aspects/location/policy resolution logic is shared via
`_resolve_listing_data()` (returns a `_ResolvedListingData`) between `create_draft_listing()` and
`update_draft_listing()`, since both need the exact same steps — only what happens with the result (POST a
new offer vs. PUT an update to an existing one) differs. Order: category resolution (see below) ->
`resolve_condition()` -> `resolve_aspects()` -> `get_merchant_location_key()` via the Location API ->
`get_listing_policies()` via the Account API (each of the latter two independently swallows failures and
returns `None`/`{}` rather than raising, since neither is required to create a valid draft, only to
*publish* one).

**Category resolution prefers an already-confirmed `category_id` over re-guessing.**
`_resolve_listing_data()` takes an optional `category_id_override` — when given, it skips
`suggest_category_id()` (the free-text Taxonomy guess) entirely and uses it directly.
`create_draft_listing()` always passes `identification.category_id` here, since by the time a draft is
created the user has already confirmed a real category via `/api/identify/confirm` (see `app.py` above).
`update_draft_listing()` is more nuanced: if the caller passes an explicit `category_query` (the user is
changing category at the *later* review screen), that's treated as an override-in-progress —
`category_id_override=None` so it re-suggests from the new text; otherwise it keeps
`identification.category_id` as before. `suggest_categories()` (used by `/api/identify` and
`/api/categories/search` in `app.py`) is the multi-result sibling of `suggest_category_id()` — same
`get_category_suggestions` endpoint, but returns the top N as `CategorySuggestion` (id + full breadcrumb
name built from `categoryTreeNodeAncestors`, confirmed live to be root-to-leaf-ordered) instead of just the
first match, so the frontend can offer a dropdown of real, valid eBay categories rather than trusting one
guess.

`create_draft_listing()`: `_resolve_listing_data()` -> `createOrReplaceInventoryItem` (PUT) ->
`createOffer` (POST, `format: "FIXED_PRICE"`, omitting `categoryId`/`merchantLocationKey`/`listingPolicies`
entirely when not found, never sending them as `null`). `update_draft_listing()`: same
`_resolve_listing_data()` call -> `createOrReplaceInventoryItem` again (same SKU — idempotent PUT, updates
the existing inventory item in place) -> `update_offer()` (PUT to `/sell/inventory/v1/offer/{offerId}`, same
`offer_id` — eBay's updateOffer is a full-replacement PUT like createOrReplaceInventoryItem, not a partial
patch, so the whole payload is rebuilt fresh via the same `_build_offer_payload()` rather than merged with
the prior offer state; safe to change `categoryId` here specifically because the offer is still
unpublished — this is a different, unsupported operation on an already-*published* listing, which this
codebase doesn't attempt). Both functions return the same `DraftListingResult` shape (`included`/`missing`/
`notes`/`aspects`/`category_query`), so the frontend's review form can be redrawn identically after either a
create or a save.

**Required item specifics ("aspects") are resolved per category, not just Brand.** Many eBay categories
reject listing creation outright if required aspects are missing — e.g. Headphones requires
Brand/Model/Type/Connectivity/Color, not just Brand — surfacing as `errorId 25002` ("item specific X is
missing"), discovered from a real production error (the same error family as the merchant-location
"Item.Country" case). `get_required_aspects()` fetches these via the Taxonomy API's
`get_item_aspects_for_category`; `resolve_aspects()` fills them from `identification` data already
available, in order: (1) direct field match for Brand/Model, (2) a **word-boundary** match (not naive
substring — a naive check for shoe size `"9"` false-positived inside `"Air Max 90"` during testing, since
fixed with a regex `\b` boundary) of one of eBay's suggested values against the identification's own text,
(3) one of eBay's own "unknown" catch-all values when the aspect offers one (e.g. `"Unbranded"`, `"Not
Applicable"` — these appear in eBay's own suggested-values lists, so using them is never a fabrication), (4)
for `FREE_TEXT`-mode aspects only (eBay's suggested values are autocomplete, not a strict enum, so any string
is accepted) an honest `"Not Specified"` placeholder, surfaced in the frontend preview as a flagged
auto-fill. Only a genuinely unresolvable non-free-text aspect (no data match, no catch-all value — e.g.
"Department" for a shoe category, since guessing a gender department would be a real fabrication) raises a
`RuntimeError` and stops `create_draft_listing()` **before** calling eBay at all — unlike
merchant_location/listing_policies (deferred to publish-time and fixable via account setup), item specifics
are validated by eBay when the offer/inventory item is created, so letting it through would just fail
moments later with the same cryptic error this exists to avoid; failing fast with our own clear message
saves the round-trip. This also can't be worked around via Seller Hub the way location/policies gaps can —
Seller Hub has no view into an unpublished offer at all (see the Seller Hub finding above), so pointing the
user there for an item-specifics gap would be actively wrong.

**Condition resolution is category-aware, not a flat static mapping — this was a real bug, not a
hypothetical.** `_CONDITION_MAP` (our 6-tier `ProductIdentification.condition` -> a default `ConditionEnum`
guess) is only a fallback now. eBay categories restrict which conditions are actually valid — e.g. Headphones
only allows New/Open-box/Used/For-parts, not the full 6-tier scale — and sending a disallowed one fails with
`errorId 25021` ("invalid for the selected primary category id"), discovered from a real production error.
`resolve_condition()` calls the Metadata API's `get_item_condition_policies` for the resolved category and
picks the closest allowed `ConditionEnum` via `_CONDITION_ENUM_PREFERENCE` (e.g. "Very Good"/"Good" both
collapse to `USED_EXCELLENT` for a category with only one generic "Used" tier), falling back to the static
guess — and adding a `notes` entry so the user knows to double-check — only when nothing in the category's
allowed set matches any of our candidates. **Empirically confirmed finding:** `get_item_condition_policies`'s
`filter=categoryIds:X` query param is silently ignored (returns the entire ~15k-entry marketplace catalog
instead of filtering) unless the value is wrapped in braces — `filter=categoryIds:{X}` — which isn't
documented anywhere obvious; found by testing directly against production rather than trusting the docs.

`publish_offer(config, token, offer_id)` is a separate, single-purpose function — the only place in this
codebase that calls eBay's `publishOffer` endpoint (`POST /sell/inventory/v1/offer/{offerId}/publish/`, no
request body). It's a **real, consequential write**: the listing becomes live and publicly purchasable
immediately, not a reversible "draft-publish" state. It's only ever reached via `POST
/api/listing/publish/{offer_id}` in `app.py`, which is only ever called from the frontend's explicit,
checkbox-gated "Publish to eBay" button — never automatically, never as part of `create_draft_listing()`.
Grep for `publishOffer`/`publish_offer` as a guardrail check before merging any change that touches this
file — it should appear in exactly this one function and its call sites, nowhere implicit.

**`src/ebay/`** — three pieces. **User OAuth 2.0** (authorization-code grant): `config.py`'s
`EbayConfig`/`load_ebay_config()` resolves sandbox vs. production base/token/authorize URLs, and
`DEFAULT_SCOPES` (`sell.inventory` + `sell.account.readonly`); `oauth.py` builds the consent URL and does the
`authorization_code`/`refresh_token` exchanges against eBay's Identity API (`EbayTokens` tracks both tokens'
expiry; eBay doesn't rotate the refresh token on refresh); `token_store.py` persists tokens to git-ignored
`data/ebay_tokens.json` (single-user, local/small deployment — not a multi-tenant design) and exposes
`get_valid_access_token()`, which transparently refreshes an expired access token before returning it;
`src/web/ebay_routes.py` wires this into FastAPI: `GET /ebay/connect` (redirect to eBay, with CSRF `state`),
`GET /ebay/callback` (code -> tokens), `GET /ebay/status`. **Application-level Browse API access** (separate,
`browse.py`): `config.py`'s `EbayBrowseConfig`/`load_ebay_browse_config()`, always production, no user
consent involved. **Listing creation** (`listing.py`, above) reuses the User OAuth token via
`get_valid_access_token()` — it's the only piece of `src/ebay/` that actually writes to eBay.

**Package layout is deliberate:** `src/`, `src/ml/`, `src/agents/`, `src/web/`, `src/ebay/` have no
`__init__.py` — they work as Python 3.14 implicit namespace packages. Do not add empty `__init__.py` files
back in; only add one if it needs to hold real code.

### Planned architecture (not yet built)

Per the original design: a hierarchical Orchestrator-Workers pattern where a Claude-based orchestrator
routes between the Vision Subagent, pricing, and draft-listing creation (all of which exist now as
independent pieces wired together by the frontend's button-by-button flow, not by an orchestrator). Pricing
(`browse.py`) and listing creation (`listing.py`) are both intentionally plain eBay API + Python, not
orchestrated subagents — no LLM reasoning involved in either. The in-app preview-then-publish flow described
above **is** this project's concrete v1 Human-in-the-Loop gate for eBay writes: eBay's own Seller Hub cannot
serve this purpose (see the Seller Hub finding at the top of this file), so the review happens in this app's
own UI instead of a manual step in eBay's UI. A more general orchestrator-level HITL gate (e.g. an explicit
human-approval step before *creating* a draft, not just before publishing one) is still future work if the
project's scope grows to need it.

**Known limitations, accepted rather than solved:**
- No TTL/cleanup for abandoned uploads in `data/uploads/` (see `app.py` note above).
- Condition validity is now checked dynamically per category (`resolve_condition()`, see above) rather than
  trusting a static mapping, but `_CONDITION_ID_TO_ENUM` only covers the standard 1000-7000 conditionId
  sequence — category groups with additional non-standard IDs (fashion categories' 2990/3010 for "Pre-owned -
  Excellent/Fair", observed live but not independently confirmed) fall through to the best-effort guess and
  a `notes` warning rather than a confirmed match. Category itself is no longer a single best-effort guess —
  the user confirms a real Taxonomy suggestion (or re-searches) up front (see `app.py`'s `/api/identify`
  above) — but the *later* edit-before-publish screen's category-search-terms field still drives a fresh
  `suggest_category_id()` free-text guess rather than the same dropdown-of-real-suggestions UX; picking a
  wrong category there still just surfaces eBay's rejection cleanly via `_call_ebay()` rather than
  pre-validating.
- The Business Policies "opt-in" step (Seller Hub only, not API-exposed) and whether eBay's sandbox even
  supports it are both outside this codebase's control — `get_listing_policies()` degrades gracefully either
  way.
- The frontend's publish confirmation is a single checkbox, not e.g. a typed confirmation phrase or a
  second server-side confirmation round-trip — considered sufficient friction for this project's scale, but
  worth revisiting if this ever handles a real seller's actual inventory at volume.
