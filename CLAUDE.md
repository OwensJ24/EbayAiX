# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AgentX (repo name `EbayAiX`) is a portfolio project demonstrating production-grade AI agent architecture: an
e-commerce helper that ingests photos of a physical item, identifies it, prices it, and drafts a live eBay
listing. It's built to showcase Agent Architecture Design, Computer Vision (via Claude's vision analysis),
Agentic Search, and AI Governance (Human-in-the-Loop controls) as resume-relevant skills, so code should
reflect deliberate, enterprise-style patterns rather than the shortest path to a working demo.

So far: batch photo upload (up to 60 photos in one session, spanning as many physical items as the user is
listing at once) with an interactive split-divider UI for the user to manually group photos by item, a menu
screen to pick one item at a time, a two-phase Claude-based structured identification per item (a cheap
name+category guess, confirmed/corrected by a human, then a fuller analysis informed by that confirmation and
by eBay's own required item-specifics for the confirmed category), a low-cost eBay comparable-listings lookup
(no LLM involved), real eBay listing creation via the legacy Trading API (`AddFixedPriceItem`, with every one
of that item's uploaded photos attached, up to eBay's real 24-photo-per-listing limit), an in-app listing
preview + explicit publish step, a FastAPI front end chaining all of it together, and an eBay OAuth 2.0
connect flow. The Claude-based orchestrator tying these steps together programmatically, and a more general
HITL approval gate, are still future work (see "Planned architecture") — but both the mandatory
identify/confirm step and the in-app review-then-publish flow already built are this project's concrete
instances of that gate.

**Item grouping is manual, not AI-driven — this was tried the other way and reverted per explicit
direction.** An earlier version of this feature sent the whole photo batch to Claude in one call
(`cluster_items()`, since removed) to automatically group photos by physical item. It didn't group reliably
enough to trust — a real, observed failure mode, not a hypothetical — so it was replaced with the "Interactive
Split Dividers" pattern in `index.html`: photos load in a single sequential strip, in the order the user added
them, and the user taps the space between any two photos to toggle an item boundary there. This is both
simpler (no clustering call, no need to validate/repair an LLM's grouping output) and more reliable (the
human, not a model, decides where one item's photos end and the next begins) — a deliberate trade of one more
manual step for correctness the AI approach couldn't reach.

**Important finding that shaped the publish flow, and later reshaped it again:** this project originally
created listings via eBay's newer Inventory API (`createOrReplaceInventoryItem` + `createOffer` +
`publishOffer`). That approach hit two real problems, in order. First: eBay's Seller Hub has no UI at all for
reviewing an unpublished offer created via the Inventory API — confirmed via multiple independent eBay
developer community threads, not a bug or misconfiguration on this project's side. (An earlier version of
this project linked to a Seller Hub "drafts" URL after creating a draft; that link was based on an incorrect
assumption and was removed.) Because eBay provided no native review surface for API-created offers, this
app's own frontend became the Human-in-the-Loop review point — showing a full preview of everything that
would go into the listing before an explicit, checkbox-gated user action made it live. Second, and more
serious: **listings created via the Inventory API could not be edited in eBay's own mobile app afterward**
— confirmed by the user's own real-world testing, not a hypothetical. This is what actually forced the
migration to the Trading API (`src/ebay/listing.py`, below), which has no such editability restriction.
Trading API's `AddFixedPriceItem` has no draft/unpublished state at all — a successful call goes immediately
live — so the pre-publish review screen is now **entirely local** (no eBay object exists until the explicit
"Publish" click); this is a *stronger* HITL guarantee than the old design, not a weaker one, since literally
nothing is sent to eBay until that one click.

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

# Vision Subagent CLI, standalone (single local image, no human confirmation step —
# chains preview() straight into identify() using preview's own guess as "confirmed")
uv run python -m src.agents.vision_subagent <image_path> [--model claude-sonnet-5]

# Web app (upload UI + API), served at http://127.0.0.1:8000
uv run uvicorn src.web.app:app --reload
```

`ANTHROPIC_API_KEY` must be set in `.env` (see `.env.example`) for anything in `src/agents/` to work —
loaded via `python-dotenv`.

`EBAY_APP_ID`, `EBAY_CERT_ID`, `EBAY_RU_NAME` (and `EBAY_ENVIRONMENT`, default `sandbox`) must be set in
`.env` for the OAuth connect flow to work. `EBAY_RU_NAME` is not a URL — it's the RuName string eBay
generates for a registered redirect URI. Separately, `EBAY_PROD_APP_ID`/`EBAY_PROD_CERT_ID` (your
**production** keyset, not sandbox) must be set for `src/ebay/browse.py`'s comp search — see .env.example
and the Architecture section below for why this specifically needs production credentials.

`SUPABASE_URL`, `SUPABASE_SECRET_KEY`, `SUPABASE_BUCKET` must be set for image uploads to work at all
(see `src/storage/supabase_storage.py` in Architecture below) — use the **secret** key (`sb_secret_...`,
Settings -> API Keys -> Secret keys), Supabase's modern replacement for the legacy service_role key (used the
same way; legacy service_role keys still work but are slated for deprecation by end of 2026). Not the
publishable/anon key, since this is a server-side-only integration that needs elevated write access. The
bucket must be configured as **public** in the Supabase dashboard (Storage -> the bucket -> "Public bucket"
toggle) or eBay won't be able to fetch images without authentication. This replaces the old
`PUBLIC_BASE_URL`-based approach
of serving images from this app's own `/uploads` mount, which no longer exists.

**If a previously-connected eBay account stops seeing business policies/location/category suggestions in
draft listings**, it's because the OAuth scope was widened after that connection was made
(`sell.account.readonly`, then later the base `https://api.ebay.com/oauth/api_scope` needed for the Taxonomy
API's category suggestions — see `config.py`'s `DEFAULT_SCOPES`) — the stored refresh token doesn't
retroactively gain scopes. Fix: click "Connect eBay account" again.

## Deployment

`Dockerfile` builds the whole app (FastAPI) as one image, targeting Render (see `render.yaml`) — chosen over
serverless platforms (Vercel, etc.) because the file-based token/location storage (`token_store.py`,
`seller_location.py`) doesn't fit a stateless serverless model. Key details baked into the Dockerfile:

- **`ENV MALLOC_ARENA_MAX=1`** caps glibc to a single malloc arena, which reliably releases freed memory back
  to the OS (unlike the multi-arena default, a well-known cause of RSS ratcheting upward across requests
  rather than resetting) — worthwhile on Render's free-tier 512MB limit given this app can process up to
  MAX_BATCH_IMAGES (60) photos per identify request (Pillow resize/convert + one base64-encoded Claude
  `preview()` call per user-defined item group — see the Architecture section below).
- **uvicorn is started with `--proxy-headers --forwarded-allow-ips='*'`.** Render terminates TLS at its edge
  and forwards plain HTTP internally, so without trusting `X-Forwarded-Proto`, `request.base_url` (used by
  `_public_base_url()` in `app.py` to build the `imageUrls` eBay fetches) silently resolves to `http://`
  instead of `https://` — uvicorn's default trusted-proxy list is only `127.0.0.1`, which doesn't match
  Render's actual internal proxy IP. This was a real, confirmed bug: eBay requires HTTPS image URLs and
  silently drops non-HTTPS ones (no error — the listing just publishes with a broken image), which is
  exactly what was happening. `--forwarded-allow-ips='*'` is safe specifically because this
  container has no other direct public exposure — the only thing that can reach it is Render's own edge.
  Confirmed via local repro (`curl -H "X-Forwarded-Proto: https"` against the built image) both before and
  after adding the flag. If image issues ever recur, setting `PUBLIC_BASE_URL` explicitly sidesteps this
  proxy-trust question entirely, at the cost of one more env var to keep in sync with the deployed domain.
- Test locally with `docker build -t agentx-test .` then `docker run -p 8001:8000 --env-file .env agentx-test`.

**eBay's OAuth redirect requires real HTTPS** — it will not redirect to a `localhost` URL no matter how
the RuName's "Auth accepted URL" is configured (eBay silently shows its own generic confirmation page
instead). This is one of the reasons the app needs a real HTTPS deployment rather than only running locally.

## Architecture

**`src/agents/vision_subagent.py`** — calls the Claude API, in two deliberate phases with a mandatory human
checkpoint between them, per explicit direction: name and category affect everything downstream (comparable-
listings search, eBay category-specific condition/aspect rules, buyer discoverability), so they're confirmed
*first* rather than being just another guess the user might fix later. There's no local ML pre-pass anymore
(an earlier local ResNet50 classifier stage was removed entirely per explicit direction, judged unnecessary
overhead — Claude's own vision analysis is what actually drives identification); both phases take one or
more photos directly.

`VisionSubagent.preview(images: list[tuple[bytes, str]]) -> ItemPreview` — a cheap first pass (`max_tokens=256`)
that returns only `item_name` + a free-text `category` guess. `images` is a list of `(image_bytes, media_type)`
pairs, each turned into its own `{"type": "image", ...}` content block in a single Claude message — Claude
supports several images per message natively, so multiple angles of the same item (including a tag or label
photo) are examined together rather than one at a time. `POST /api/identify` (see `app.py` below) calls this
once per item group the user defines via the split-divider UI — one call per group, not one call over the
whole batch — using the first `MAX_ANALYSIS_IMAGES` of that group's photos; the result seeds that item's menu
card (name + category) and is never used directly for listing creation.

`VisionSubagent.identify(images, confirmed_item_name, confirmed_category, required_aspects=None) ->
ProductIdentification` — the fuller analysis, only ever run *after* a human has confirmed/edited the name and
picked a real eBay category (see `app.py`'s `/api/identify/confirm`). `confirmed_item_name`/`confirmed_category`
are required params, not an optional override — the system prompt tells Claude to treat them as ground truth
and analyze everything else (brand, model number, condition, description); after parsing, `identify()`
defensively overwrites `item_name`/`category` on the result with the confirmed strings rather than trusting
the model to echo them back exactly. `required_aspects` (eBay's required item specifics for the confirmed
category, from `listing.py`'s `get_required_aspects()` — see `app.py`'s `/api/identify/confirm` below) is
optional but always passed in practice; when given, the prompt explicitly lists what eBay requires (e.g.
"Brand, Model, Type, Connectivity, Color") and asks Claude to look across all provided photos — including
tags/labels/packaging — for each one, using eBay's own suggested vocabulary where it matches. This directly
improves `resolve_aspects()`'s word-boundary text-matching in `listing.py` (unchanged code, richer input)
rather than requiring any change to that matching logic itself. Uses structured outputs
(`client.messages.parse(..., output_format=ProductIdentification)`) to force a strict schema: item name,
brand, model number, category, `category_id` (the confirmed real eBay categoryId, set by the caller —
Claude never produces this), a constrained condition enum, condition notes, a buyer-facing
`content_description`, distinguishing features, and a confidence level. Be conservative about condition:
only claim 'New' if there is clear evidence (tags, packaging, no wear); return `null` for brand/model number
rather than guess.

**`content_description` vs. `condition_notes` — deliberately separate, per explicit direction.**
`content_description` is the *body* of the eBay listing description — `_build_description()` in
`listing.py` prepends `identification.item_name` as a title line above it (never trusting Claude to
reproduce the title verbatim; the title is added programmatically so it can never drift from whatever the
user actually confirmed/edited). Per explicit direction, `content_description` itself is written like a real,
simple eBay description, not narrative prose: (1) a short functionality-confirmation line when applicable —
"Tested and working" for a functional/electronic item, or an honest non-functional equivalent (e.g. "Sold for
parts, not tested") when condition is `"For Parts"` or otherwise non-functional, omitted entirely for items
where a working/non-working claim doesn't apply (clothing, books, decorative items) — then (2) brief,
plain-fact specifics: model number, style, size, material, color, included accessories. The system prompt
explicitly tells Claude not to mention wear, damage, or condition there. `condition_notes` stays as
condition-rating justification only (and still feeds `resolve_aspects()`'s text-matching corpus in
`listing.py`) — it's never blended into the buyer-facing description.

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

**`src/web/app.py`** — the FastAPI front end. `POST /api/identify` accepts a whole *batch* of photos in one
request — up to `MAX_BATCH_IMAGES` (60), already split into per-item groups by the user (the frontend's
split-divider UI — see below) before this route is ever called
(`files: list[UploadFile]` — FastAPI's native multi-file support, matched by repeated `files` fields in the
frontend's `FormData`; the field name must match the parameter name exactly, since that's how FastAPI maps
multipart fields — a real bug hit during development when the frontend used `'file'` instead — plus a
`group_sizes: str` form field, a JSON-encoded array of positive integers like `"[3, 2, 4]"` giving each item's
photo count, in the same order as `files`; validated to sum to exactly the number of uploaded files). Every
file is read, extension-allowlisted (jpg/jpeg/png/webp — all formats eBay accepts, confirmed against eBay's
own docs), size-capped at 10MB, and validated/normalized *entirely in memory* via
`_validate_and_normalize_image()` — rejects anything under `EBAY_MIN_IMAGE_DIMENSION` (500px) on either side
with a clear 400 (eBay's own docs say a smaller image "may be blocked" from the listing, silently, which is a
worse failure mode than rejecting it up front), and **downscales anything over `EBAY_MAX_IMAGE_DIMENSION`
(1600px, matching eBay's own "preferably at least 1600x1600" guidance) to 1600px on its longest side, always
re-encoding as JPEG regardless of original format.** This exists because full-resolution modern phone photos
(4000px+, several MB), multiplied by up to `MAX_BATCH_IMAGES` per upload, were plausibly large enough to push
a request past Render's free-tier timeout or memory limit — the observed symptom was a bare 502 with **zero**
matching Render log output (the request died mid-flight, before this app's own logging/error handling ever
ran), a much less diagnosable failure mode than this app's normal 4xx/502/503 JSON responses. *Every* file is
validated/normalized before *any* of them are uploaded, so one bad photo aborts the whole request cleanly
rather than leaving a partial set in Supabase. Each normalized image is then uploaded to **Supabase Storage**
(`src/storage/supabase_storage.py`, see below) at `{upload_id}/{index}.jpg`.

**Grouping is the user's own input, not something this route figures out.** Since splits only ever divide the
sequential upload order (the user can't reorder photos, only mark boundaries between them), each item's
`photo_indices` is always a contiguous range — `group_sizes` is walked to build them
(`[3, 2]` -> item 0 gets `[0,1,2]`, item 1 gets `[3,4]`). For each item, one `VisionSubagent.preview()` call
(see `vision_subagent.py` above) runs over that item's first `MAX_ANALYSIS_IMAGES` photos to seed its menu
card with a real name/category guess — one cheap call per item, not one large call over the whole batch,
which is both simpler and, in aggregate, usually cheaper than the AI-clustering approach this replaced (that
one call had to examine the entire batch at once; N small calls each only look at one item's own handful of
photos). Returns `{status: "batch_preview", upload_id, image_count, items: [{item_index, photo_indices,
item_name, category}, ...]}` — `item_index` is just each item's 0-based position in this list, but it matters
later: since several items now share one `upload_id`, it's what keeps their eBay SKUs from colliding at
publish time (see `listing.py`'s `generate_sku()` below). No Taxonomy call happens here — category
suggestions are fetched lazily, only once the user actually opens a given item (see
`POST /api/categories/search` below), since not every item in a batch necessarily gets worked on in the same
sitting.

**`POST /api/categories/search`** — the same `suggest_categories()` call `/api/identify` used to run inline
for its single item before batch upload existed; now reused for two things: the frontend's re-search box when
none of the initial suggestions fit, *and* the first thing that happens when a user opens an item from the
batch menu (using that item's rough `category` guess from its `preview()` call as the query, to populate its
category dropdown).

**`POST /api/identify/confirm`** (this used to be `/api/identify/refine`, with a materially different
contract — renamed since it's no longer "refine a low-confidence guess" but "run the full analysis now that
name+category are confirmed"). Body: `{upload_id, item_name, category_id, category_name, photo_indices}` —
all mandatory, never optional. `photo_indices` identifies *this item's* photos within the shared batch (in
practice always a contiguous range — see the split-divider grouping above — but the field itself is a plain
index list, not a range, so nothing downstream needs to assume contiguity); since this is a separate HTTP
request from `/api/identify` with
no server-side session state, it re-fetches the first `photo_indices[:MAX_ANALYSIS_IMAGES]` via `httpx.get()`
on their already-uploaded Supabase public URLs (same pattern `create_listing()` in `listing.py` uses to
re-fetch an image before EPS upload) rather than keeping anything in memory across requests. It also fetches
this category's required item specifics via `get_required_aspects()` (`listing.py`, reused as-is, using the
same application-level Taxonomy token) *before* calling `VisionSubagent.identify()` (Phase 2 — the full
analysis, informed by the *confirmed* name/category and by what eBay actually requires for this category,
rather than re-guessing either), sets `identification.category_id = category_id` on the result (Claude never
produces this — it's the real eBay Taxonomy ID the user picked), and returns the final `ProductIdentification`.
Every downstream step (`/api/price`, `/api/listing/draft`, the edit-before-publish screen,
`/api/listing/publish/{upload_id}`) is unchanged — they already just consume whatever `ProductIdentification`
is in the frontend's `currentIdentification`.

**Image lifecycle: no local disk staging at all — Supabase Storage is the only durable store.** Removing the
local ResNet50 classifier (which needed a real file path — see `vision_subagent.py` above) removed the only
reason this app ever kept a local temp file: every remaining consumer (Claude's vision calls) only ever
needed image *bytes*. Photos now go straight from upload -> in-memory validation/normalization -> Supabase,
for any number of photos, with nothing ever written to local disk. `public_url(config, upload_id, index)`
reconstructs each Supabase image URL purely from the deterministic `{upload_id}/{index}.jpg` naming, no
existence check needed, since `upload_image()` already succeeded synchronously back in `/api/identify` (if
that failed, the flow never got this far).

**The Supabase objects are never automatically deleted, even after a successful publish — this was tried and
reverted after a real recurrence of the "image not available" bug.** The first version of this integration
deleted the image right after `publish_listing_route()`'s `create_listing()` call returned, on the reasoning
that publish success meant eBay was done needing it. That reasoning was wrong: a successful publish doesn't
mean eBay has actually *fetched* the `PictureURL` yet — that fetch is lazy/asynchronous, the exact same
behavior that caused the original pre-Supabase version of this bug — so deleting immediately after publish
recreated the same race condition one layer further down the pipeline. This finding predates (and is
independent of) the later Trading API migration below — it applies equally to `AddFixedPriceItem`'s
`PictureURL` field(s). Confirmed the Supabase upload and public-URL fetch both work correctly in isolation (a
live round-trip test: upload, then an unauthenticated GET exactly simulating eBay's fetch, returned the
correct bytes and content-type) before concluding the premature deletion was the actual cause. Fix: don't
delete anything. A user who abandons the flow, or a published listing, both leave permanent Supabase objects;
no cleanup job exists for this (`delete_images()` in `supabase_storage.py` exists as a usable utility for a
future manual cleanup script, but nothing currently calls it).

Once the confirm step (above) returns a final `ProductIdentification`, the frontend automatically posts it
(FastAPI validates the JSON body directly against that Pydantic model) to
`POST /api/price`, which calls `search_comparable_listings()` above using a query built by
`browse.build_query()` (shared with the Taxonomy category lookup in `listing.py`, see below). The average of
the returned comps pre-fills an editable price field. The user must also enter a package weight (lbs) — a
real, blocking eBay requirement (`packageWeightAndSize.weight`, `errorId 25020` "package weight is not valid
or is missing" if omitted, discovered from a real production error — the Vision Subagent has no way to
estimate this from a photo, so it's the one piece of listing data the human always has to supply directly
rather than something the pipeline can infer). A "Create Draft Listing" button posts
`{identification, upload_id}` to `POST /api/listing/draft`, which calls `resolve_draft_listing()` (see
`src/ebay/listing.py` below) — a **read-only** dry run (category/condition/aspects/policy resolution only,
no eBay write of any kind), since Trading API's `AddFixedPriceItem` (used at publish time) has no
draft/unpublished state to create here at all.

Once that resolution completes, the frontend shows an **editable** review form (title, brand, condition
dropdown, eBay category search terms, description, plus the existing price/weight inputs), pre-filled from
the same `ProductIdentification` already in memory, plus any `missing`/`notes` gaps and resolved `aspects`
from the result. This form — not eBay's Seller Hub, and not any object on eBay's side, since none exists yet
— is the human review *and correction* point: the automatic identification/category-suggestion pipeline can
be wrong, so every field that matters for a correct listing (title, brand, condition, category, description)
is editable here, not just visible. A "Save Changes" button posts the edited fields to
`POST /api/listing/draft/{upload_id}/update`, which re-runs `resolve_draft_listing()` with the edited fields
— again purely a read-only re-resolution, not an eBay write; `upload_id` in the path is for URL readability
only; nothing is looked up by it. The edited description textarea becomes the new `condition_notes` directly
(with `distinguishing_features` cleared) rather than being reconstructed from separate fields, so a second
edit doesn't duplicate content. Publishing requires checking an explicit confirmation checkbox ("I understand
this will create a real, live eBay listing...") before the "Publish to eBay" button enables; clicking it
builds a fresh `ProductIdentification` from whatever is currently shown in the edit form (not necessarily
whatever was last saved via "Save Changes" — publish always reflects the current on-screen state) and posts
it, along with price/weight/currency, to `POST /api/listing/publish/{upload_id}`, which calls
`create_listing()` (see `src/ebay/listing.py` below) — **the one and only point in the entire flow that
writes anything to eBay** — and returns a real, public `listing_url` on success.

**The confirmation checkbox is disabled outright (not just a warning) whenever `result.missing` is
non-empty.** `category`/`location`/`listing_policies` are each hard requirements for a successful
`AddFixedPriceItem` call, so letting the user attempt to publish anyway just wastes a round-trip to eBay for
a guaranteed failure. The `location` requirement traces back to a real production error from this project's
prior Inventory-API design: `publishOffer` returned `errorId 25002` with the message `"No <Item.Country>
exists..."`, which sounds like a missing address field but actually meant no inventory location was set up on
the eBay account — confirmed via eBay developer community threads, the same kind of misleadingly-worded eBay
error as the earlier `Content-Language`/25709 case. The Trading API migration below eliminated the underlying
eBay-side location object entirely, but the same category of gap (nothing configured yet) still needs to
block publish, so the check and its `missing`/`notes` plumbing carry over unchanged in shape.

**`POST /api/ebay/location`** (frontend: a collapsed "Set up shipping location" `<details>` block above the
upload dropzone, now just `postal_code` + `country`) calls `save_seller_location()`
(`src/ebay/seller_location.py`) — a plain local JSON file write (`data/seller_location.json`, git-ignored,
same persistence style as `token_store.py`'s `data/ebay_tokens.json`), **not an eBay API call at all.**
Trading API's `AddFixedPriceItem` needs only flat `Item.Country`/`Item.PostalCode` fields on the request
itself; unlike the Inventory API's `merchantLocationKey` system (registered once via the Location API, then
looked up on every draft), there's no eBay-side location object to register or query — `load_seller_location()`
just reads the same local file back at publish time. Meant to be run once per deployment (the file is
git-ignored local state, not per-eBay-account).

The `VisionSubagent` instance is constructed once at module import time and reused across requests —
re-instantiating per request would recreate the Anthropic client every call. `_call_claude()`
wraps every Claude-backed call and `_call_ebay()` wraps every eBay-backed call, each translating their
respective API failures into clean `4xx`/`502`/`503` HTTP responses instead of a bare 500 —
`_call_claude()` was added after a real Anthropic-side outage surfaced as an unhandled exception during
development; `_call_ebay()` additionally catches the `RuntimeError` that `load_ebay_config()`/
`load_ebay_browse_config()`/`get_valid_access_token()` raise for missing env vars or a not-yet-connected
eBay account — this used to reach the client as a bare 500 before `_call_ebay()` existed.

**`_call_claude()` also catches `anthropic.APIResponseValidationError` explicitly, plus a catch-all
`Exception` fallback, both logged via `logger.exception()`.** Found by inspecting the `anthropic` package's
exception hierarchy directly: `APIResponseValidationError` (raised when Claude's response fails to validate
against the requested structured-output schema — a real, if uncommon, failure mode for `output_format=...`
calls) is a sibling of `APIStatusError` under the base `APIError`, not a subclass of it, so it silently
escaped the original four `except` clauses as an opaque, untraceable 500. The catch-all exists for the same
reason — any future exception type from a Claude call (e.g. a raw `pydantic.ValidationError`) now gets
logged with a full traceback and a specific error message instead of vanishing into a generic crash.

**`src/storage/supabase_storage.py`** — hosts uploaded item photos on Supabase Storage instead of serving
them from this app's own Render instance. Raw `httpx` calls to Supabase's Storage REST API, matching this
project's eBay integration style (no SDK dependency): `upload_image(config, upload_id, index, contents)`
(`POST /storage/v1/object/{bucket}/{path}` with `apikey` + `Authorization: Bearer` headers both set to the
secret key, plus `x-upsert: true` so re-running identify for the same `upload_id`/`index` overwrites cleanly
rather than erroring), `public_url(config, upload_id, index)` (pure string construction from
`{SUPABASE_URL}/storage/v1/object/public/{bucket}/{upload_id}/{index}.jpg` — deterministic, never checks
existence), and `delete_images(config, upload_id, image_count)` (`DELETE /storage/v1/object/{bucket}` with a
JSON `{"prefixes": [...]}` body listing every `{upload_id}/{i}.jpg` path — Supabase's batch-delete shape,
confirmed against Supabase's own API docs rather than assumed; best-effort, logs failures instead of
raising). All of a *batch's* photos — potentially spanning several items, see `app.py` above — live under the
same shared `{upload_id}/` prefix, one object per index, with a fully deterministic key scheme; which indices
belong to which item is tracked entirely in the frontend's `currentBatch` state (each item's `photo_indices`),
not by this module. Both `upload_image()` and `public_url()` log their request/response and the final URL via
`logger.info()` — this was added specifically to debug a real "image not available" recurrence (see the note
on `publish_listing_route()` above); `delete_images()` exists as a usable utility (e.g. for a future manual
cleanup script) but **nothing in this codebase currently calls it** — see below for why. **Why Supabase and
not this app's own `/uploads` endpoint** (the prior design, now
removed along with `PUBLIC_BASE_URL`): Render's free tier spins down after inactivity, but eBay fetches
`imageUrls` lazily rather than synchronously when a draft is created — so the old design had a real
reliability gap where eBay's fetch could land while this app's own instance was asleep. Supabase Storage has
no such sleep state, so the image is reliably fetchable independent of this app's own uptime. The **bucket
must be manually configured as public** in the Supabase dashboard (Storage -> the bucket -> "Public bucket"
toggle) — there's no API to set this, and an eBay fetch against a private bucket's URL would just fail with
an auth error, which would surface as the same "image not available" symptom this change exists to prevent.

**`src/ebay/listing.py`** — resolves everything a listing needs (read-only), then makes exactly one real
eBay write via the legacy **Trading API**. This is the file that was rewritten wholesale for the Inventory
API → Trading API migration described at the top of this file; everything downstream of category/condition/
aspects/policy resolution — the actual listing-creation mechanics — is new. The shared resolution logic
lives in `_resolve_listing_data()` (returns a `_ResolvedListingData`), used by both `resolve_draft_listing()`
(the read-only pre-publish preview/edit dry run) and `create_listing()` (the one real write). Order: category
resolution (see below) -> `resolve_condition()` -> `resolve_aspects()` -> `load_seller_location()` (a plain
local file read, not an eBay call — see the `POST /api/ebay/location` note above) -> `get_listing_policies()`
via the Account API (the latter two independently degrade to `None`/`{}` rather than raising, since neither
is required to *resolve* a listing, only to *publish* one).

**Category resolution prefers an already-confirmed `category_id` over re-guessing.**
`_resolve_listing_data()` takes an optional `category_id_override` — when given, it skips
`suggest_category_id()` (the free-text Taxonomy guess) entirely and uses it directly. `create_listing()`
always passes `identification.category_id` here, since by the time publish happens the user has already
confirmed a real category via `/api/identify/confirm` (see `app.py` above). `resolve_draft_listing()` is more
nuanced: if the caller passes an explicit `category_query` (the user is changing category at the *later*
review screen), that's treated as an override-in-progress — `category_id_override=None` so it re-suggests
from the new text; otherwise it keeps `identification.category_id` as before. `suggest_categories()` (used by
`/api/identify` and `/api/categories/search` in `app.py`) is the multi-result sibling of
`suggest_category_id()` — same `get_category_suggestions` endpoint, but returns the top N as
`CategorySuggestion` (id + full breadcrumb name built from `categoryTreeNodeAncestors`, confirmed live to be
root-to-leaf-ordered) instead of just the first match, so the frontend can offer a dropdown of real, valid
eBay categories rather than trusting one guess.

`resolve_draft_listing()`: just `_resolve_listing_data()`, returning a `DraftListingResult`
(`included`/`missing`/`notes`/`aspects`/`category_query`) — no eBay write, called from both
`POST /api/listing/draft` and `POST /api/listing/draft/{upload_id}/update` in `app.py` (there's no separate
create-vs-update distinction anymore, since there's nothing on eBay to create or update yet; both routes just
re-run the same dry run against whatever `ProductIdentification` the frontend currently has). `create_listing()`
takes `upload_id` *and* `item_index` — since one `upload_id` can now be a whole batch shared across several
items (see `app.py` above), `item_index` (that item's 0-based position in `/api/identify`'s response) is what
`generate_sku(upload_id, item_index) -> f"agentx-{upload_id}-{item_index}"` uses to keep two items published
from the same batch from colliding on the same eBay SKU. For every photo in `image_urls` (already sliced to
`MAX_ITEM_IMAGES` by `app.py`'s publish route as a defensive cap), fetches the Supabase-hosted bytes and
re-hosts them on eBay's own Picture Services via `upload_site_hosted_picture()` (see below) — sequentially,
not in parallel, matching this codebase's style elsewhere, at the cost of publish latency roughly proportional
to photo count — then `_resolve_listing_data()` -> `_build_add_fixed_price_item_request()` (builds the full
`AddFixedPriceItemRequest` XML tree via `xml.etree.ElementTree`'s `Element`/`SubElement` API — **deliberately
not hand-rolled string templates**, since real titles/categories contain `&`/`<`/`>` (already seen in this
project's own data, e.g. "Portable Audio & Headphones") that a string template would silently turn into
invalid XML; ElementTree escapes correctly for free, verified live — with one `PictureDetails/PictureURL`
child element per photo, within eBay's confirmed real limit of 24 self-hosted `PictureURL`s per listing —
not the ~12-per-*variation* figure an earlier version of this doc incorrectly generalized from) ->
`_call_trading_api(..., "AddFixedPriceItem", ...)` -> extracts `ItemID` from the response and returns a
`CreateListingResult` (`item_id`, `listing_url`, `missing`, `notes`).

**Required item specifics ("aspects") are resolved per category, not just Brand — and now actively targeted
during identification, not just matched after the fact.** Many eBay categories reject listing creation
outright if required aspects are missing — e.g. Headphones requires
Brand/Model/Type/Connectivity/Color, not just Brand — surfacing as `errorId 25002` ("item specific X is
missing"), discovered from a real production error under the prior Inventory API design (the same error
family as the old merchant-location "Item.Country" case). `get_required_aspects()` fetches these via the
Taxonomy API's `get_item_aspects_for_category`; besides its use here, `app.py`'s `/api/identify/confirm` now
also calls it *before* `VisionSubagent.identify()` and passes the result in as `identify()`'s
`required_aspects` param (see `vision_subagent.py` above), so Claude actively looks for each required aspect
across all provided photos during analysis — rather than this function's `resolve_aspects()` being the only
place trying to recover them, after the fact, from whatever free-text Claude happened to write.
`resolve_aspects()` fills them from `identification` data
already available, in order: (1) direct field match for Brand/Model, (2) a **word-boundary** match (not naive
substring — a naive check for shoe size `"9"` false-positived inside `"Air Max 90"` during testing, since
fixed with a regex `\b` boundary) of one of eBay's suggested values against the identification's own text,
(3) one of eBay's own "unknown" catch-all values when the aspect offers one (e.g. `"Unbranded"`, `"Not
Applicable"` — these appear in eBay's own suggested-values lists, so using them is never a fabrication), (4)
for `FREE_TEXT`-mode aspects only (eBay's suggested values are autocomplete, not a strict enum, so any string
is accepted) an honest `"Not Specified"` placeholder, surfaced in the frontend preview as a flagged
auto-fill. Only a genuinely unresolvable non-free-text aspect (no data match, no catch-all value — e.g.
"Department" for a shoe category, since guessing a gender department would be a real fabrication) raises a
`RuntimeError` from `_resolve_listing_data()` — surfaced to the user via the read-only draft-resolution
routes well before any eBay write is attempted, rather than failing at `AddFixedPriceItem` time with the same
cryptic error this exists to avoid.

**Condition resolution is category-aware, not a flat static mapping — this was a real bug, not a
hypothetical.** `_CONDITION_MAP` (our 6-tier `ProductIdentification.condition` -> a default numeric
`ConditionID` guess) is only a fallback now. eBay categories restrict which conditions are actually valid —
e.g. Headphones only allows New/Open-box/Used/For-parts, not the full 6-tier scale — and sending a disallowed
one fails with `errorId 25021` ("invalid for the selected primary category id"), discovered from a real
production error. `resolve_condition()` calls the Metadata API's `get_item_condition_policies` for the
resolved category and picks the closest allowed `ConditionID` via `_CONDITION_ID_PREFERENCE` (e.g. "Very
Good"/"Good" both prefer `4000` for a category that allows it), falling back to the static guess — and adding
a `notes` entry so the user knows to double-check — only when nothing in the category's allowed set matches
any of our candidates. Returns the numeric ID directly now, not a translated enum string — Trading API's
`Item.ConditionID` field takes eBay's legacy numeric condition IDs as-is, so the REST Inventory API's
`ConditionEnum` translation layer this project used before is gone entirely. **Empirically confirmed
finding:** `get_item_condition_policies`'s `filter=categoryIds:X` query param is silently ignored (returns the
entire ~15k-entry marketplace catalog instead of filtering) unless the value is wrapped in braces —
`filter=categoryIds:{X}` — which isn't documented anywhere obvious; found by testing directly against
production rather than trusting the docs.

**Trading API's error signaling is fundamentally different from the REST APIs used elsewhere in this
codebase, and had to be handled explicitly.** `AddFixedPriceItem` (like all Trading API calls) returns
**HTTP 200 even when the call fails at the application level** — the real success/failure signal is the
response body's `<Ack>Success|Warning|Failure</Ack>` element, with details in `<Errors>`. `_call_trading_api()`
parses this explicitly and raises `EbayTradingApiError` on `Ack in ("Failure", "PartialFailure")`; a plain
`response.raise_for_status()` (which this project's REST calls rely on) would silently treat a rejected
listing as a success. `_call_ebay()` in `app.py` has a dedicated `except EbayTradingApiError` clause (checked
before the generic `httpx.HTTPStatusError` clause, since Trading API failures never raise that) that turns it
into a clean `502`. Auth is also different: Trading API takes the same user OAuth access token this app
already stores, but via an `X-EBAY-API-IAF-TOKEN` header rather than REST's `Authorization: Bearer` — see
`_call_trading_api()`.

`create_listing()` is the **only** place in this codebase that calls eBay's `AddFixedPriceItem` — a **real,
consequential write**: the listing becomes live and publicly purchasable immediately, with no reversible
"draft" state beforehand (unlike the old Inventory API's separate create-then-`publishOffer` step). It's only
ever reached via `POST /api/listing/publish/{upload_id}` in `app.py`, which is only ever called from the
frontend's explicit, checkbox-gated "Publish to eBay" button — never automatically, never as a side effect of
`resolve_draft_listing()`. Grep for `AddFixedPriceItem`/`create_listing` as a guardrail check before merging
any change that touches this file — it should appear in exactly this one function and its call sites, nowhere
implicit.

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
consent involved. **Listing creation** (`listing.py`, above) reuses the same User OAuth access token via
`get_valid_access_token()` — it's the only piece of `src/ebay/` that actually writes to eBay — but sends it
via Trading API's `X-EBAY-API-IAF-TOKEN` header rather than the `Authorization: Bearer` header the REST calls
elsewhere in this file use; no changes to `oauth.py`/`token_store.py` were needed for this, since it's the
same token, just carried differently on the wire.

**Package layout is deliberate:** `src/`, `src/agents/`, `src/web/`, `src/ebay/` have no `__init__.py` — they
work as Python 3.14 implicit namespace packages. Do not add empty `__init__.py` files back in; only add one
if it needs to hold real code.

### Planned architecture (not yet built)

Per the original design: a hierarchical Orchestrator-Workers pattern where a Claude-based orchestrator
routes between the Vision Subagent, pricing, and draft-listing creation (all of which exist now as
independent pieces wired together by the frontend's button-by-button flow, not by an orchestrator). Pricing
(`browse.py`) and listing creation (`listing.py`) are both intentionally plain eBay API + Python, not
orchestrated subagents — no LLM reasoning involved in either. The in-app preview-then-publish flow described
above **is** this project's concrete v1 Human-in-the-Loop gate for eBay writes — and is now a *stronger* gate
than it started out as: there isn't merely no Seller Hub view into it (the original reasoning, back when
listings were drafted via the Inventory API), there's no eBay-side object of any kind until the explicit
Publish click, since Trading API's `AddFixedPriceItem` has no unpublished state to create early. A more
general orchestrator-level HITL gate (e.g. an explicit human-approval step somewhere earlier in the pipeline,
not just before the one eBay write) is still future work if the project's scope grows to need it.

**Known limitations, accepted rather than solved:**
- No TTL/cleanup for abandoned uploads left in Supabase Storage (see `app.py`/`supabase_storage.py` notes
  above) — there's no local disk equivalent of this problem anymore (no local staging at all, see the Image
  lifecycle note above), but the Supabase side still has no cleanup job.
- `create_listing()`'s per-photo EPS re-hosting loop (see `listing.py` above) is sequential, not parallel, so
  publish latency scales roughly linearly with photo count (up to `MAX_ITEM_IMAGES` = 24) — acceptable given
  this project's scale, and consistent with the rest of the codebase never using concurrent HTTP calls, but
  worth revisiting (e.g. `httpx.AsyncClient` + `asyncio.gather`) if publish latency becomes a real UX problem.
- The whole batch — the split-divider grouping, each item's `photo_indices`, and every item's `status` (not
  started/in progress/published) — lives entirely in the frontend's `currentBatch` JS state, with no
  server-side session store. A page refresh loses the whole batch, requiring a full re-upload and re-split;
  this was already true for a single item before this feature, just now scaled up to potentially several
  items' worth of in-progress work.
- Condition validity is checked dynamically per category (`resolve_condition()`, see above) rather than
  trusting a static mapping, but `_CONDITION_ID_PREFERENCE` only covers the standard 1000-7000 conditionId
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
- This app has no way to *edit* a listing after publish (e.g. a `ReviseFixedPriceItem`-backed "edit" feature)
  — by design, since the entire point of this migration was to make listings editable via eBay's own native
  mobile app instead.
- `_TRADING_API_VERSION` (the `X-EBAY-API-COMPATIBILITY-LEVEL` header value) is a hardcoded string that should
  be checked against eBay's Trading API release notes periodically; it isn't dynamically discovered.
