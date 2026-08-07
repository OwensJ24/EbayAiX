"""eBay Trading API: create a real, live listing in a single call.

Flow: best-effort category/condition/aspects/policy resolution (all read-only REST
Taxonomy/Metadata/Account API calls, no eBay writes) -> a single AddFixedPriceItem
call, the only write in this module.

**Why the Trading API and not the newer Inventory API this project used before:**
listings created via the Inventory API (createOrReplaceInventoryItem + createOffer +
publishOffer) cannot be edited in eBay's own mobile app afterward — confirmed by real
account testing, not a hypothetical. The Trading API doesn't have this limitation.

**Why there's no "draft" object on eBay anymore:** unlike createOffer/publishOffer's
two-step create-then-publish, Trading API's AddFixedPriceItem has no unpublished state
at all — a successful call goes immediately live (confirmed against eBay's own docs and
developer community, not assumed). Since eBay's Seller Hub already couldn't show
Inventory-API drafts either, this app's own frontend was already the real
Human-in-the-Loop review surface — that doesn't change. What changes is that the
review/edit screen is now entirely local (no eBay object exists to create or update
before publish); the explicit, checkbox-gated Publish click is the one and only time
this module ever writes to eBay at all. That's a *stronger* safety guarantee than
before, not a weaker one.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel

from src.agents.vision_subagent import ProductIdentification
from src.ebay.browse import build_query
from src.ebay.config import EbayConfig, load_ebay_config
from src.ebay.seller_location import SellerLocation, load_seller_location
from src.ebay.token_store import get_valid_access_token

logger = logging.getLogger(__name__)

_MARKETPLACE_ID = "EBAY_US"
_CATEGORY_TREE_ID = "0"  # EBAY_US

# eBay's legacy numeric ConditionID — this IS what Trading API's Item.ConditionID field
# wants directly, no enum translation layer needed (unlike the REST Inventory API's
# ConditionEnum strings this project used before). Confirmed against eBay's own
# condition-ID reference and community sources.
_CONDITION_MAP: dict[str, str] = {
    "New": "1000",
    "Like New": "2750",
    "Very Good": "4000",
    "Good": "5000",
    "Acceptable": "6000",
    "For Parts": "7000",
}

# For each of our own condition labels, ConditionID candidates in closest-first order —
# used to pick the nearest eBay-allowed condition for a category when _CONDITION_MAP's
# default guess isn't one of that category's allowed conditions (categories restrict
# which IDs are valid — e.g. Headphones only allows New/Open-box/Used/For-parts, not
# our full 6-tier scale — sending a disallowed one fails with errorId 25021).
_CONDITION_ID_PREFERENCE: dict[str, tuple[str, ...]] = {
    "New": ("1000", "2000", "1500"),
    "Like New": ("2750", "3000", "1500", "2500"),
    "Very Good": ("4000", "3000", "2750"),
    "Good": ("5000", "4000", "3000"),
    "Acceptable": ("6000", "5000", "4000", "3000"),
    "For Parts": ("7000", "6000"),
}


def _auth_headers(token: str) -> dict[str, str]:
    # Used only for the REST Taxonomy/Metadata/Account calls below (category
    # suggestions, condition policies, required aspects, business policies) — all
    # read-only, independent of Inventory vs. Trading API. Content-Language is required
    # on every Sell API call that writes data; harmless on GETs, so applied uniformly.
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Content-Language": "en-US",
        "Accept-Language": "en-US",
    }


def generate_sku(upload_id: str, item_index: int) -> str:
    # item_index disambiguates SKUs when several items are published from the same
    # upload_id — a single batch upload (see app.py's /api/identify) can hold multiple
    # items' photos, each published independently, and eBay requires unique SKUs.
    return f"agentx-{upload_id}-{item_index}"


def _build_description(identification: ProductIdentification) -> str:
    # Title first (matches how real eBay listing descriptions are typically written),
    # then the Claude-written body — a functionality confirmation line where applicable,
    # followed by brief specifics. Built here rather than by Claude so the title can
    # never drift from whatever the user actually confirmed/edited. Deliberately excludes
    # condition_notes — condition/wear commentary has its own dedicated field and UI.
    body = identification.content_description.strip()
    full_description = f"{identification.item_name}\n\n{body}" if body else identification.item_name
    return full_description[:4000]


def suggest_category_id(config: EbayConfig, token: str, query: str) -> str | None:
    try:
        response = httpx.get(
            f"{config.api_base}/commerce/taxonomy/v1/category_tree/{_CATEGORY_TREE_ID}/get_category_suggestions",
            headers=_auth_headers(token),
            params={"q": query},
            timeout=15.0,
        )
        response.raise_for_status()
        suggestions = response.json().get("categorySuggestions", [])
        if not suggestions:
            logger.info("suggest_category_id degraded: no suggestions returned for query=%r", query)
            return None
        return suggestions[0]["category"]["categoryId"]
    except httpx.HTTPStatusError as e:
        logger.info("suggest_category_id degraded: %d %s", e.response.status_code, e.response.text[:300])
        return None
    except (httpx.HTTPError, KeyError, IndexError) as e:
        logger.info("suggest_category_id degraded: %r", e)
        return None


class CategorySuggestion(BaseModel):
    category_id: str
    category_name: str


def _breadcrumb(suggestion: dict) -> str:
    ancestors = suggestion.get("categoryTreeNodeAncestors", [])
    # eBay returns ancestors nearest-parent-first; reverse for a root-to-leaf breadcrumb.
    names = [a["categoryName"] for a in reversed(ancestors)]
    names.append(suggestion["category"]["categoryName"])
    return " > ".join(names)


def suggest_categories(config: EbayConfig, token: str, query: str, limit: int = 5) -> list[CategorySuggestion]:
    """Like suggest_category_id(), but returns the top `limit` ranked suggestions with
    full breadcrumb names instead of just the first categoryId — used so the user can
    pick from a dropdown of real, valid eBay categories rather than trusting a single
    auto-guess or typing free text. Degrades to an empty list on any failure, same as
    suggest_category_id()'s None.
    """
    try:
        response = httpx.get(
            f"{config.api_base}/commerce/taxonomy/v1/category_tree/{_CATEGORY_TREE_ID}/get_category_suggestions",
            headers=_auth_headers(token),
            params={"q": query},
            timeout=15.0,
        )
        response.raise_for_status()
        suggestions = response.json().get("categorySuggestions", [])
        return [
            CategorySuggestion(category_id=s["category"]["categoryId"], category_name=_breadcrumb(s))
            for s in suggestions[:limit]
        ]
    except httpx.HTTPStatusError as e:
        logger.info("suggest_categories degraded: %d %s", e.response.status_code, e.response.text[:300])
        return []
    except (httpx.HTTPError, KeyError, IndexError) as e:
        logger.info("suggest_categories degraded: %r", e)
        return []


def resolve_condition(config: EbayConfig, token: str, category_id: str | None, condition: str) -> tuple[str, bool]:
    """Picks a numeric eBay ConditionID for `condition` that eBay's Metadata API
    confirms is valid for `category_id`. Returns (condition_id, confirmed) — falls back
    to the static _CONDITION_MAP guess with confirmed=False when there's no category,
    the lookup fails, or none of our own candidates are in that category's allowed set
    (e.g. fashion categories using non-standard conditionIds like 2990/3010 we don't
    have in our preference table).
    """
    default = _CONDITION_MAP[condition]
    if not category_id:
        return default, False

    try:
        response = httpx.get(
            f"{config.api_base}/sell/metadata/v1/marketplace/{_MARKETPLACE_ID}/get_item_condition_policies",
            headers=_auth_headers(token),
            # eBay's filter expression requires braces around the value — confirmed
            # empirically: `categoryIds:123` is silently ignored (returns the entire
            # ~15k-entry marketplace catalog instead of filtering); only
            # `categoryIds:{123}` actually filters to the requested category.
            params={"filter": f"categoryIds:{{{category_id}}}"},
            timeout=15.0,
        )
        response.raise_for_status()
        policies = response.json().get("itemConditionPolicies", [])
        if not policies:
            logger.info("resolve_condition degraded: no condition policy for category=%s", category_id)
            return default, False

        allowed_ids = {c["conditionId"] for c in policies[0].get("itemConditions", [])}
        for candidate in _CONDITION_ID_PREFERENCE[condition]:
            if candidate in allowed_ids:
                return candidate, True

        logger.info(
            "resolve_condition degraded: none of %s allowed for category=%s (allowed ids=%s)",
            _CONDITION_ID_PREFERENCE[condition],
            category_id,
            allowed_ids,
        )
        return default, False
    except (httpx.HTTPError, KeyError, IndexError) as e:
        logger.info("resolve_condition degraded: %r", e)
        return default, False


def get_category_aspects(config: EbayConfig, token: str, category_id: str) -> list[dict]:
    """Fetches eBay's item specifics ('aspects') for a category via the Taxonomy API —
    both hard-required aspects and ones eBay marks RECOMMENDED (e.g. for Video Games &
    Consoles: Publisher, Genre, Rating, Region Code, Release Year are all RECOMMENDED,
    not required, and were previously filtered out entirely — a real, reported gap, not
    a hypothetical). Skips OPTIONAL-usage aspects (e.g. "California Prop 65 Warning"):
    low-value enough that attempting to fill them mostly adds noise or fabrication risk
    for little buyer-facing benefit.

    Each returned dict has `name`, `values` (eBay's suggested values — an autocomplete
    list, not a strict enum, when `mode` is "FREE_TEXT"), `mode`, and `required` (bool).
    `required` comes from `aspectConstraint.aspectRequired`, NOT `aspectUsage` — eBay's
    own docs confirm `aspectUsage` is unreliable for this specifically, since a truly
    required aspect is *also* reported as `aspectUsage: "RECOMMENDED"` (confirmed live
    against a real category's response); `aspectRequired` is the only trustworthy
    signal for whether an unresolved aspect should actually block listing creation (see
    resolve_aspects() below) — required aspects still do; recommended ones don't, they
    just get left out of the listing if genuinely undeterminable, e.g. errorId 25002
    "item specific X is missing" only applies to the former.
    """
    try:
        response = httpx.get(
            f"{config.api_base}/commerce/taxonomy/v1/category_tree/{_CATEGORY_TREE_ID}/get_item_aspects_for_category",
            headers=_auth_headers(token),
            params={"category_id": category_id},
            timeout=15.0,
        )
        response.raise_for_status()
        aspects = []
        for aspect in response.json().get("aspects", []):
            constraint = aspect.get("aspectConstraint", {})
            required = bool(constraint.get("aspectRequired"))
            if not required and constraint.get("aspectUsage") != "RECOMMENDED":
                continue
            aspects.append(
                {
                    "name": aspect["localizedAspectName"],
                    "values": [
                        v["localizedValue"] for v in aspect.get("aspectValues", []) if v.get("localizedValue")
                    ],
                    "mode": constraint.get("aspectMode"),
                    "required": required,
                }
            )
        return aspects
    except (httpx.HTTPError, KeyError) as e:
        logger.info("get_category_aspects degraded: %r", e)
        return []


# eBay's own suggested-values lists for otherwise-unknown required aspects commonly
# include a designated catch-all (e.g. "Unbranded", "Not Applicable") — preferring one
# of these over inventing a value keeps every aspect we send eBay-sanctioned and honest.
_UNKNOWN_ASPECT_MARKERS = ("not applicable", "does not apply", "unbranded", "unknown", "n/a", "no brand")


def _find_fallback_aspect_value(values: list[str]) -> str | None:
    return next((v for v in values if v.strip().lower() in _UNKNOWN_ASPECT_MARKERS), None)


def resolve_aspects(
    identification: ProductIdentification, category_aspects: list[dict]
) -> tuple[dict[str, list[str]], list[str]]:
    """Best-effort fills eBay's item specifics from data already in `identification` —
    never fabricates a plausible-sounding but made-up value (e.g. never guesses a
    color). Resolution order per aspect: (1) direct field match for Brand/Model, (2) a
    substring match of one of eBay's suggested values against the identification's own
    text, (3) one of eBay's own "unknown" catch-all values if it offers one, (4) for
    FREE_TEXT aspects only (no real enum to violate) an honest 'Not Specified'
    placeholder. Only a genuinely unresolvable aspect with `required=True` ends up in
    the returned `unresolved` list — an unresolvable *recommended* aspect (see
    get_category_aspects()) is simply left out of the returned dict instead, since
    eBay doesn't need it to accept the listing and blocking publish over an optional
    field like "Genre" would be wrong.
    """
    corpus = " ".join(
        [
            identification.item_name,
            identification.condition_notes,
            identification.content_description,
            *identification.distinguishing_features,
        ]
    ).lower()

    aspects: dict[str, list[str]] = {}
    unresolved: list[str] = []

    for aspect in category_aspects:
        name = aspect["name"]
        values = aspect["values"]

        if name == "Brand" and identification.brand:
            aspects[name] = [identification.brand]
            continue
        if name in ("Model", "MPN", "Manufacturer Part Number") and identification.model_number:
            aspects[name] = [identification.model_number]
            continue

        # Word-boundary match, not plain substring containment — e.g. a naive `in`
        # check for shoe size "9" would false-positive inside "Air Max 90".
        match = next(
            (v for v in values if re.search(rf"\b{re.escape(v.lower())}\b", corpus)), None
        )
        if match:
            aspects[name] = [match]
            continue

        fallback = _find_fallback_aspect_value(values)
        if fallback:
            aspects[name] = [fallback]
            continue

        if aspect["mode"] == "FREE_TEXT":
            aspects[name] = ["Not Specified"]
            continue

        if aspect.get("required"):
            unresolved.append(name)
        # else: unresolvable but only recommended, not required — leave it out rather
        # than block or fabricate.

    return aspects, unresolved


_POLICY_ENDPOINTS = {
    "fulfillmentPolicyId": ("fulfillment_policy", "fulfillmentPolicies", "fulfillmentPolicyId"),
    "paymentPolicyId": ("payment_policy", "paymentPolicies", "paymentPolicyId"),
    "returnPolicyId": ("return_policy", "returnPolicies", "returnPolicyId"),
}


def get_listing_policies(config: EbayConfig, token: str) -> dict[str, str]:
    policies: dict[str, str] = {}
    for offer_key, (endpoint, list_key, id_key) in _POLICY_ENDPOINTS.items():
        try:
            response = httpx.get(
                f"{config.api_base}/sell/account/v1/{endpoint}",
                headers=_auth_headers(token),
                params={"marketplace_id": _MARKETPLACE_ID},
                timeout=15.0,
            )
            response.raise_for_status()
            items = response.json().get(list_key, [])
            if not items:
                continue
            if offer_key == "paymentPolicyId":
                # Every listing this app creates has Best Offer enabled (see
                # _build_add_fixed_price_item_request()), and eBay's own platform
                # rules make Best Offer and "require immediate payment" mutually
                # exclusive on a single listing — confirmed from a real rejection
                # (errorId 23015, "If this item sells by a Best Offer, you will not
                # be able to require immediate payment"). Prefer a payment policy
                # that doesn't require immediate payment over the account's
                # first/default one, which may well be an immediate-pay policy set
                # up before this app ever enabled Best Offer. Falls back to the
                # first policy if every one of the account's payment policies
                # requires immediate payment — this app has no way to change a
                # payment policy's own settings remotely, so at that point eBay's
                # own error (surfaced cleanly via EbayTradingApiError) is what tells
                # the seller to fix it in Seller Hub.
                chosen = next((p for p in items if not p.get("immediatePay")), items[0])
                policies[offer_key] = chosen[id_key]
            else:
                policies[offer_key] = items[0][id_key]
        except httpx.HTTPStatusError as e:
            logger.info("get_listing_policies(%s) degraded: %d %s", endpoint, e.response.status_code, e.response.text[:300])
        except (httpx.HTTPError, KeyError, IndexError) as e:
            logger.info("get_listing_policies(%s) degraded: %r", endpoint, e)
    return policies


class _ResolvedListingData(BaseModel):
    category_id: str | None
    condition_id: str
    aspects: dict[str, list[str]]
    listing_policies: dict[str, str]
    included: list[str]
    missing: list[str]
    notes: list[str]


def _resolve_listing_data(
    config: EbayConfig,
    token: str,
    identification: ProductIdentification,
    query: str,
    category_id_override: str | None = None,
) -> _ResolvedListingData:
    """Shared by the pre-publish preview/edit routes and the final publish call:
    resolves category/condition/aspects/policies from `identification` and the given
    category search query. Every call this function makes is a read-only REST GET
    (Taxonomy/Metadata/Account APIs) — it never writes anything to eBay. Raises
    RuntimeError for a genuinely unresolvable required aspect (see resolve_aspects) —
    fails fast rather than letting a guaranteed eBay rejection happen at publish time.

    `category_id_override`, when given, skips the Taxonomy suggestion call entirely —
    used when the user already confirmed a real category via the identify/confirm step
    (or hasn't changed it in the later edit screen), so there's no need to re-guess.
    """
    included: list[str] = []
    missing: list[str] = []
    notes: list[str] = []

    if category_id_override:
        category_id = category_id_override
        included.append("category")
    else:
        category_id = suggest_category_id(config, token, query)
        if category_id:
            included.append("category")
        else:
            missing.append("category")
            notes.append("No eBay category suggestion found — try more specific category search terms.")

    condition_id, condition_confirmed = resolve_condition(config, token, category_id, identification.condition)
    if category_id and not condition_confirmed:
        notes.append(
            f"Couldn't confirm '{identification.condition}' is a valid eBay condition for the detected "
            f"category — used a best-effort guess; double-check it before publishing."
        )

    aspects: dict[str, list[str]] = {}
    if category_id:
        category_aspects = get_category_aspects(config, token, category_id)
        aspects, unresolved_aspects = resolve_aspects(identification, category_aspects)
        if unresolved_aspects:
            # eBay validates item specifics when the listing is created — letting this
            # through would just fail moments later with the same cryptic errorId 25002
            # this whole function exists to avoid. Fail fast with a clear message
            # instead of spending an eBay round-trip on a guaranteed rejection.
            raise RuntimeError(
                "eBay requires a value for these item specifics in the detected category, and it "
                f"couldn't be determined automatically: {', '.join(unresolved_aspects)}. Try editing the "
                "title/description with more detail, or a more specific category search."
            )
    elif identification.brand:
        aspects = {"Brand": [identification.brand]}

    listing_policies = get_listing_policies(config, token)
    if len(listing_policies) == 3:
        included.append("listing_policies")
    else:
        missing.append("listing_policies")
        notes.append(
            "Payment/fulfillment/return business policies aren't fully set up (or this "
            "eBay connection predates the sell.account.readonly scope — reconnect via "
            "/ebay/connect to enable policy detection). Add them in Seller Hub before publishing."
        )

    if load_seller_location() is None:
        missing.append("location")
        notes.append("No shipping location saved yet — set one up above before publishing.")
    else:
        included.append("location")

    return _ResolvedListingData(
        category_id=category_id,
        condition_id=condition_id,
        aspects=aspects,
        listing_policies=listing_policies,
        included=included,
        missing=missing,
        notes=notes,
    )


class DraftListingResult(BaseModel):
    included: list[str]
    missing: list[str]
    notes: list[str]
    aspects: dict[str, list[str]] = {}
    category_query: str = ""


def resolve_draft_listing(
    identification: ProductIdentification,
    category_query: str | None = None,
) -> DraftListingResult:
    """The pre-publish preview/edit step — resolves category/condition/aspects/policies
    and reports what's included vs. missing, exactly like before, but calls nothing on
    eBay that writes data. There's no draft object to create or update anymore (Trading
    API's AddFixedPriceItem has no unpublished state — see module docstring), so this
    is purely a read-only dry run that feeds the review screen.
    """
    config = load_ebay_config()
    token = get_valid_access_token(config)

    if category_query:
        query = category_query
        category_id_override = None
    else:
        query = build_query(identification)
        category_id_override = identification.category_id

    data = _resolve_listing_data(config, token, identification, query, category_id_override=category_id_override)

    return DraftListingResult(
        included=data.included,
        missing=data.missing,
        notes=data.notes,
        aspects=data.aspects,
        category_query=query,
    )


def _listing_url(config: EbayConfig, item_id: str) -> str:
    base = "https://www.sandbox.ebay.com" if config.environment == "sandbox" else "https://www.ebay.com"
    return f"{base}/itm/{item_id}"


_EB_NS_URI = "urn:ebay:apis:eBLBaseComponents"
_NS = {"eb": _EB_NS_URI}
# eBay's Trading API release version this app builds requests against — check
# https://developer.ebay.com/devzone/xml/docs/ReleaseNotes.html for the current value
# and bump this if a call starts failing with a version-related error.
_TRADING_API_VERSION = "1155"
_SITE_ID = "0"  # EBAY_US


class EbayTradingApiError(Exception):
    """Raised when a Trading API call returns Ack=Failure (or an applicable Warning).
    Trading API returns HTTP 200 even for these — the real success/failure signal is
    the <Ack> element in the response body, not the HTTP status code, so
    response.raise_for_status() alone would silently treat a rejected listing as a
    success.
    """

    def __init__(self, errors: list[dict[str, str | None]]):
        self.errors = errors
        message = "; ".join(f"[{e.get('code')}] {e.get('message')}" for e in errors) or "eBay Trading API call failed"
        super().__init__(message)


def _sub(parent: ET.Element, tag: str, text: str | None = None) -> ET.Element:
    """SubElement shorthand. ElementTree escapes text content automatically — unlike a
    hand-rolled string template, which would silently produce invalid XML for titles or
    category names containing &, <, or > (all seen in this project's own real data,
    e.g. "Portable Audio & Headphones")."""
    el = ET.SubElement(parent, tag)
    if text is not None:
        el.text = text
    return el


def _check_ack(root: ET.Element) -> None:
    """Shared by every Trading API call (plain-XML and multipart alike): raises
    EbayTradingApiError on Ack=Failure/PartialFailure — see EbayTradingApiError's
    docstring for why this can't just be response.raise_for_status()."""
    ack = root.findtext("eb:Ack", namespaces=_NS)
    if ack in ("Failure", "PartialFailure"):
        errors = [
            {
                "code": err.findtext("eb:ErrorCode", namespaces=_NS),
                "message": (
                    err.findtext("eb:LongMessage", namespaces=_NS)
                    or err.findtext("eb:ShortMessage", namespaces=_NS)
                ),
            }
            for err in root.findall("eb:Errors", namespaces=_NS)
        ]
        raise EbayTradingApiError(errors)


def _trading_api_headers(token: str, call_name: str) -> dict[str, str]:
    return {
        "X-EBAY-API-SITEID": _SITE_ID,
        "X-EBAY-API-COMPATIBILITY-LEVEL": _TRADING_API_VERSION,
        "X-EBAY-API-CALL-NAME": call_name,
        "X-EBAY-API-IAF-TOKEN": token,
    }


def _call_trading_api(config: EbayConfig, token: str, call_name: str, request_root: ET.Element) -> ET.Element:
    # encoding="utf-8" (rather than "unicode") makes ET.tostring prepend the
    # <?xml version='1.0' encoding='utf-8'?> declaration — required by eBay's Trading
    # API gateway. Without it, eBay's internal XML->SOAP translation layer breaks and
    # surfaces a generic, misleading SAXParseException about "soapenv:Body" instead of
    # any error actually describing the missing declaration (confirmed live).
    body = ET.tostring(request_root, encoding="utf-8", xml_declaration=True)
    headers = {**_trading_api_headers(token, call_name), "Content-Type": "text/xml"}
    logger.info(
        "%s request: url=%s headers=%s body=%s",
        call_name,
        f"{config.api_base}/ws/api.dll",
        {k: v for k, v in headers.items() if k != "X-EBAY-API-IAF-TOKEN"},
        body.decode("utf-8"),
    )
    response = httpx.post(
        f"{config.api_base}/ws/api.dll",
        headers=headers,
        content=body,
        timeout=30.0,
    )
    logger.info("%s response: status=%d body=%s", call_name, response.status_code, response.text[:3000])
    response.raise_for_status()  # still catches real HTTP/network-level failures

    root = ET.fromstring(response.text)
    _check_ack(root)
    return root


def upload_site_hosted_picture(config: EbayConfig, token: str, image_bytes: bytes, picture_name: str) -> str:
    """Uploads image bytes directly to eBay Picture Services (EPS) via the Trading
    API's UploadSiteHostedPictures call, returning the resulting EPS-hosted FullURL
    (an https://i.ebayimg.com/... URL).

    Used instead of pointing AddFixedPriceItem's PictureDetails/PictureURL at the
    Supabase-hosted image directly: eBay's backend auto-mirrors self-hosted
    PictureURLs into EPS behind the scenes, and when that mirroring is inconsistent it
    fails the listing with errorId 20004 "A mixture of Self Hosted and EPS pictures
    are not allowed" — a real, recurring issue confirmed via eBay developer/community
    reports, not specific to this app's own request shape. Pre-uploading directly to
    EPS ourselves avoids that ambiguity entirely, since we then only ever reference an
    EPS URL.

    Unlike every other Trading API call in this module, this one is a MIME-multipart
    POST (XML control fields + the raw image bytes as a second part), not a plain XML
    body — eBay's docs are explicit that the binary can't be embedded in the XML
    itself (e.g. base64-inlined); it must be a separate multipart section.
    """
    request_xml = ET.Element(f"{{{_EB_NS_URI}}}UploadSiteHostedPicturesRequest")
    ET.register_namespace("", _EB_NS_URI)
    _sub(request_xml, "PictureName", picture_name)
    # "Supersize" matches the standard image sizing (incl. zoom) eBay's own web/app
    # listing photos use — the same quality level a self-hosted PictureURL implied.
    _sub(request_xml, "PictureSet", "Supersize")
    xml_body = ET.tostring(request_xml, encoding="utf-8", xml_declaration=True)

    headers = _trading_api_headers(token, "UploadSiteHostedPictures")
    logger.info(
        "UploadSiteHostedPictures request: url=%s headers=%s picture_name=%s image_bytes=%d",
        f"{config.api_base}/ws/api.dll",
        {k: v for k, v in headers.items() if k != "X-EBAY-API-IAF-TOKEN"},
        picture_name,
        len(image_bytes),
    )
    response = httpx.post(
        f"{config.api_base}/ws/api.dll",
        headers=headers,
        # httpx builds a proper multipart/form-data body (with boundary) whenever
        # `files` is given — do NOT also set a Content-Type header, or it'll clobber
        # the boundary httpx generates.
        data={"XML Payload": xml_body},
        files={"dummy": (f"{picture_name}.jpg", image_bytes, "image/jpeg")},
        timeout=60.0,
    )
    logger.info(
        "UploadSiteHostedPictures response: status=%d body=%s",
        response.status_code,
        response.text[:2000],
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    _check_ack(root)
    full_url = root.findtext(".//eb:SiteHostedPictureDetails/eb:FullURL", namespaces=_NS)
    if not full_url:
        raise EbayTradingApiError(
            [{"code": None, "message": "UploadSiteHostedPictures succeeded but returned no FullURL"}]
        )
    return full_url


def _build_add_fixed_price_item_request(
    identification: ProductIdentification,
    image_urls: list[str],
    price: float,
    weight_lbs: float,
    currency: str,
    quantity: int,
    data: _ResolvedListingData,
    location: SellerLocation,
    sku: str,
) -> ET.Element:
    ET.register_namespace("", _EB_NS_URI)
    root = ET.Element(f"{{{_EB_NS_URI}}}AddFixedPriceItemRequest")
    _sub(root, "ErrorLanguage", "en_US")
    _sub(root, "WarningLevel", "High")

    item = _sub(root, "Item")
    _sub(item, "SKU", sku)
    _sub(item, "Title", identification.item_name[:80])
    _sub(item, "Description", _build_description(identification))
    if data.category_id:
        primary_category = _sub(item, "PrimaryCategory")
        _sub(primary_category, "CategoryID", data.category_id)
    _sub(item, "ConditionID", data.condition_id)
    _sub(item, "Quantity", str(quantity))
    _sub(item, "StartPrice", f"{price:.2f}")
    _sub(item, "Currency", currency)
    _sub(item, "Country", location.country)
    _sub(item, "PostalCode", location.postal_code)
    _sub(item, "ListingDuration", "GTC")
    _sub(item, "ListingType", "FixedPriceItem")

    # Lets buyers submit a lower offer instead of paying the listed price outright, on
    # every listing this app publishes. Not every eBay category supports Best Offer
    # (confirmed against eBay's own docs) — rather than pre-checking via a separate
    # GetCategoryFeatures call for a category that will almost always support it, this
    # is sent unconditionally; the rare unsupported-category case surfaces as a normal
    # eBay rejection through the existing EbayTradingApiError handling, the same way
    # every other category-specific constraint in this file is discovered and handled.
    best_offer_details = _sub(item, "BestOfferDetails")
    _sub(best_offer_details, "BestOfferEnabled", "true")

    picture_details = _sub(item, "PictureDetails")
    for url in image_urls:
        _sub(picture_details, "PictureURL", url)

    if data.aspects:
        item_specifics = _sub(item, "ItemSpecifics")
        for name, values in data.aspects.items():
            name_value_list = _sub(item_specifics, "NameValueList")
            _sub(name_value_list, "Name", name)
            for value in values:
                _sub(name_value_list, "Value", value)

    if data.listing_policies:
        seller_profiles = _sub(item, "SellerProfiles")
        if "fulfillmentPolicyId" in data.listing_policies:
            shipping_profile = _sub(seller_profiles, "SellerShippingProfile")
            _sub(shipping_profile, "ShippingProfileID", data.listing_policies["fulfillmentPolicyId"])
        if "paymentPolicyId" in data.listing_policies:
            payment_profile = _sub(seller_profiles, "SellerPaymentProfile")
            _sub(payment_profile, "PaymentProfileID", data.listing_policies["paymentPolicyId"])
        if "returnPolicyId" in data.listing_policies:
            return_profile = _sub(seller_profiles, "SellerReturnProfile")
            _sub(return_profile, "ReturnProfileID", data.listing_policies["returnPolicyId"])

    # Trading API splits package weight into whole pounds + remainder ounces rather
    # than a single decimal-pounds value.
    shipping_package_details = _sub(item, "ShippingPackageDetails")
    _sub(shipping_package_details, "WeightMajor", str(int(weight_lbs)))
    _sub(shipping_package_details, "WeightMinor", str(round((weight_lbs % 1) * 16)))

    return root


class CreateListingResult(BaseModel):
    item_id: str
    listing_url: str
    missing: list[str]
    notes: list[str]


def create_listing(
    identification: ProductIdentification,
    upload_id: str,
    item_index: int,
    image_urls: list[str],
    price: float,
    weight_lbs: float,
    currency: str = "USD",
    quantity: int = 1,
) -> CreateListingResult:
    """The ONE eBay write in this module. Combines what used to be
    createOrReplaceInventoryItem + createOffer + publishOffer into eBay's single
    AddFixedPriceItem call. Only ever reached via the frontend's explicit,
    checkbox-gated "Publish to eBay" button — never automatically, never as a side
    effect of the pre-publish review/edit screen (see resolve_draft_listing() above,
    which never calls this). Grep for AddFixedPriceItem/create_listing as a guardrail
    check before merging any change that touches this file — it should appear in
    exactly this one function and its call sites, nowhere implicit.

    `image_urls` can be up to MAX_ITEM_IMAGES (see app.py) Supabase-hosted photos —
    every one of them is re-hosted on EPS and attached to the listing (see the loop
    below), not just a single primary photo. `upload_id` now identifies a whole batch
    of photos that can span several items (see app.py's /api/identify), so
    `item_index` — this item's position within that batch — is required for SKU
    uniqueness (see generate_sku()); two items from the same batch must never collide
    on the same SKU.
    """
    config = load_ebay_config()
    token = get_valid_access_token(config)
    sku = generate_sku(upload_id, item_index)

    location = load_seller_location()
    if location is None:
        raise RuntimeError("No shipping location saved yet — set one up before publishing.")

    # Re-host every photo on eBay's own Picture Services (EPS) rather than pointing
    # PictureDetails/PictureURL at the Supabase URLs directly — see
    # upload_site_hosted_picture()'s docstring for why self-hosted URLs are unreliable
    # here. Supabase stays the durable source of truth for the images themselves; this
    # just changes what URLs get handed to eBay for this one listing. Sequential, not
    # parallel — matches this codebase's style elsewhere, at the cost of added publish
    # latency roughly proportional to photo count.
    eps_picture_urls = []
    for i, image_url in enumerate(image_urls):
        image_response = httpx.get(image_url, timeout=30.0)
        image_response.raise_for_status()
        eps_picture_urls.append(upload_site_hosted_picture(config, token, image_response.content, f"{sku}-{i}"))

    query = build_query(identification)
    data = _resolve_listing_data(config, token, identification, query, category_id_override=identification.category_id)

    request_root = _build_add_fixed_price_item_request(
        identification, eps_picture_urls, price, weight_lbs, currency, quantity, data, location, sku
    )
    response_root = _call_trading_api(config, token, "AddFixedPriceItem", request_root)
    item_id = response_root.findtext(".//eb:ItemID", namespaces=_NS)
    if not item_id:
        raise EbayTradingApiError([{"code": None, "message": "AddFixedPriceItem succeeded but returned no ItemID"}])

    return CreateListingResult(
        item_id=item_id,
        listing_url=_listing_url(config, item_id),
        missing=data.missing,
        notes=data.notes,
    )
