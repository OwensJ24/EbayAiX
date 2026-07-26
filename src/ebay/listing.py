"""eBay Inventory + Offer APIs: create a draft listing, then optionally publish it.

Flow: createOrReplaceInventoryItem -> best-effort category/location/policy
enrichment -> createOffer -> (separate, explicit step) publishOffer.

eBay's own Seller Hub has no UI at all for reviewing an unpublished offer
created via the Inventory API (confirmed via eBay developer community
research) — so the Human-in-the-Loop safety gate for this project lives in
our own app instead: the frontend shows a full preview of everything that
will go into the listing, and only an explicit user action calls
`publish_offer()`. That function is the only place in this codebase that
calls eBay's publishOffer endpoint, and it makes the listing genuinely live
and publicly purchasable — not a reversible/staging action.
"""

from __future__ import annotations

import logging
import re

import httpx
from pydantic import BaseModel

from src.agents.vision_subagent import ProductIdentification
from src.ebay.browse import build_query
from src.ebay.config import EbayConfig, load_ebay_config
from src.ebay.token_store import get_valid_access_token

logger = logging.getLogger(__name__)

_MARKETPLACE_ID = "EBAY_US"
_CATEGORY_TREE_ID = "0"  # EBAY_US

_CONDITION_MAP: dict[str, str] = {
    "New": "NEW",
    "Like New": "LIKE_NEW",
    "Very Good": "USED_VERY_GOOD",
    "Good": "USED_GOOD",
    "Acceptable": "USED_ACCEPTABLE",
    "For Parts": "FOR_PARTS_OR_NOT_WORKING",
}

# eBay's legacy numeric conditionId (as returned by get_item_condition_policies below) ->
# the ConditionEnum string the Inventory API's `condition` field actually expects. This
# table is fixed/global — what varies per category is *which* subset of these IDs a given
# category allows (e.g. Headphones only allows New/Open-box/Used/For-parts, not our full
# 6-tier scale) — confirmed against eBay's own condition-ID reference and community
# sources. Category groups with additional non-standard IDs (fashion categories' 2990/3010
# for "Pre-owned - Excellent/Fair", observed live but not independently confirmed here)
# aren't covered — those categories fall through to the best-effort guess below.
_CONDITION_ID_TO_ENUM: dict[str, str] = {
    "1000": "NEW",
    "1500": "NEW_OTHER",
    "1750": "NEW_WITH_DEFECTS",
    "2000": "CERTIFIED_REFURBISHED",
    "2500": "SELLER_REFURBISHED",
    "2750": "LIKE_NEW",
    "3000": "USED_EXCELLENT",
    "4000": "USED_VERY_GOOD",
    "5000": "USED_GOOD",
    "6000": "USED_ACCEPTABLE",
    "7000": "FOR_PARTS_OR_NOT_WORKING",
}

# For each of our own condition labels, ConditionEnum candidates in closest-first order —
# used to pick the nearest eBay-allowed condition for a category when _CONDITION_MAP's
# default guess isn't one of that category's allowed conditions.
_CONDITION_ENUM_PREFERENCE: dict[str, tuple[str, ...]] = {
    "New": ("NEW", "CERTIFIED_REFURBISHED", "NEW_OTHER"),
    "Like New": ("LIKE_NEW", "USED_EXCELLENT", "NEW_OTHER", "SELLER_REFURBISHED"),
    "Very Good": ("USED_VERY_GOOD", "USED_EXCELLENT", "LIKE_NEW"),
    "Good": ("USED_GOOD", "USED_VERY_GOOD", "USED_EXCELLENT"),
    "Acceptable": ("USED_ACCEPTABLE", "USED_GOOD", "USED_VERY_GOOD", "USED_EXCELLENT"),
    "For Parts": ("FOR_PARTS_OR_NOT_WORKING", "USED_ACCEPTABLE"),
}


def _auth_headers(token: str) -> dict[str, str]:
    # Content-Language is required on every Sell Inventory API call that writes data
    # (createOrReplaceInventoryItem AND createOffer) — eBay's error for a *missing*
    # Content-Language header confusingly says "Invalid value for header
    # Content-Language" (error 25709) rather than something like "header required",
    # which is what sent us chasing the wrong call and the wrong fix at first.
    # Applying it to every call (including GETs, which ignore it) is simplest and safe.
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Content-Language": "en-US",
        "Accept-Language": "en-US",
    }


def generate_sku(upload_id: str) -> str:
    return f"agentx-{upload_id}"


def _build_description(identification: ProductIdentification) -> str:
    parts = [identification.condition_notes]
    if identification.distinguishing_features:
        parts.append("Notable features:")
        parts.extend(f"- {f}" for f in identification.distinguishing_features)
    description = "\n".join(parts)
    return description[:4000]


def _build_inventory_item_payload(
    identification: ProductIdentification,
    image_url: str,
    quantity: int,
    condition_enum: str,
    aspects: dict[str, list[str]],
    weight_lbs: float,
) -> dict:
    payload = {
        "condition": condition_enum,
        "product": {
            "title": identification.item_name[:80],
            "description": _build_description(identification),
            "imageUrls": [image_url],
        },
        "packageWeightAndSize": {"weight": {"value": weight_lbs, "unit": "POUND"}},
        "availability": {"shipToLocationAvailability": {"quantity": quantity}},
    }
    if aspects:
        payload["product"]["aspects"] = aspects
    return payload


def create_or_replace_inventory_item(
    config: EbayConfig,
    token: str,
    sku: str,
    identification: ProductIdentification,
    image_url: str,
    condition_enum: str,
    aspects: dict[str, list[str]],
    weight_lbs: float,
    quantity: int = 1,
) -> None:
    payload = _build_inventory_item_payload(identification, image_url, quantity, condition_enum, aspects, weight_lbs)
    headers = _auth_headers(token)

    logger.info(
        "createOrReplaceInventoryItem request: url=%s headers=%s payload=%s",
        f"{config.api_base}/sell/inventory/v1/inventory_item/{sku}",
        {k: v for k, v in headers.items() if k != "Authorization"},
        payload,
    )
    response = httpx.put(
        f"{config.api_base}/sell/inventory/v1/inventory_item/{sku}",
        headers=headers,
        json=payload,
        timeout=20.0,
    )
    logger.info(
        "createOrReplaceInventoryItem response: status=%d headers=%s body=%s",
        response.status_code,
        dict(response.headers),
        response.text,
    )
    response.raise_for_status()


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


def resolve_condition(
    config: EbayConfig, token: str, category_id: str | None, condition: str
) -> tuple[str, bool]:
    """Picks a ConditionEnum for `condition` that eBay's Metadata API confirms is valid
    for `category_id`. Returns (enum, confirmed) — falls back to the static
    _CONDITION_MAP guess with confirmed=False when there's no category, the lookup
    fails, or none of our own candidates are in that category's allowed set (e.g.
    fashion categories using non-standard conditionIds we don't have a mapping for).
    This exists because eBay categories restrict which conditions are valid — e.g.
    Headphones only allows New/Open-box/Used/For-parts, not our full 6-tier scale —
    and sending a disallowed one fails with errorId 25021 ("invalid for the selected
    primary category id").
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
        allowed_enums = {_CONDITION_ID_TO_ENUM[cid] for cid in allowed_ids if cid in _CONDITION_ID_TO_ENUM}
        for candidate in _CONDITION_ENUM_PREFERENCE[condition]:
            if candidate in allowed_enums:
                return candidate, True

        logger.info(
            "resolve_condition degraded: none of %s allowed for category=%s (allowed ids=%s)",
            _CONDITION_ENUM_PREFERENCE[condition],
            category_id,
            allowed_ids,
        )
        return default, False
    except (httpx.HTTPError, KeyError, IndexError) as e:
        logger.info("resolve_condition degraded: %r", e)
        return default, False


def get_required_aspects(config: EbayConfig, token: str, category_id: str) -> list[dict]:
    """Fetches eBay's required item specifics ('aspects') for a category via the
    Taxonomy API. Many categories reject listing creation entirely if these aren't
    present (e.g. Headphones requires Brand/Model/Type/Connectivity/Color) — this is
    what errorId 25002 "item specific X is missing" means. Each returned dict has
    `name`, `values` (eBay's suggested values — an autocomplete list, not a strict
    enum, when `mode` is "FREE_TEXT") and `mode`.
    """
    try:
        response = httpx.get(
            f"{config.api_base}/commerce/taxonomy/v1/category_tree/{_CATEGORY_TREE_ID}/get_item_aspects_for_category",
            headers=_auth_headers(token),
            params={"category_id": category_id},
            timeout=15.0,
        )
        response.raise_for_status()
        required = []
        for aspect in response.json().get("aspects", []):
            constraint = aspect.get("aspectConstraint", {})
            if not constraint.get("aspectRequired"):
                continue
            required.append(
                {
                    "name": aspect["localizedAspectName"],
                    "values": [
                        v["localizedValue"] for v in aspect.get("aspectValues", []) if v.get("localizedValue")
                    ],
                    "mode": constraint.get("aspectMode"),
                }
            )
        return required
    except (httpx.HTTPError, KeyError) as e:
        logger.info("get_required_aspects degraded: %r", e)
        return []


# eBay's own suggested-values lists for otherwise-unknown required aspects commonly
# include a designated catch-all (e.g. "Unbranded", "Not Applicable") — preferring one
# of these over inventing a value keeps every aspect we send eBay-sanctioned and honest.
_UNKNOWN_ASPECT_MARKERS = ("not applicable", "does not apply", "unbranded", "unknown", "n/a", "no brand")


def _find_fallback_aspect_value(values: list[str]) -> str | None:
    return next((v for v in values if v.strip().lower() in _UNKNOWN_ASPECT_MARKERS), None)


def resolve_aspects(
    identification: ProductIdentification, required_aspects: list[dict]
) -> tuple[dict[str, list[str]], list[str]]:
    """Best-effort fills eBay's required aspects from data already in `identification`
    — never fabricates a plausible-sounding but made-up value (e.g. never guesses a
    color). Resolution order per aspect: (1) direct field match for Brand/Model, (2) a
    substring match of one of eBay's suggested values against the identification's own
    text, (3) one of eBay's own "unknown" catch-all values if it offers one, (4) for
    FREE_TEXT aspects only (no real enum to violate) an honest 'Not Specified'
    placeholder. Only truly unresolvable non-free-text aspects end up in the returned
    unresolved list.
    """
    corpus = " ".join(
        [identification.item_name, identification.condition_notes, *identification.distinguishing_features]
    ).lower()

    aspects: dict[str, list[str]] = {}
    unresolved: list[str] = []

    for aspect in required_aspects:
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

        unresolved.append(name)

    return aspects, unresolved


def get_merchant_location_key(config: EbayConfig, token: str) -> str | None:
    try:
        response = httpx.get(
            f"{config.api_base}/sell/inventory/v1/location",
            headers=_auth_headers(token),
            params={"limit": 1},
            timeout=15.0,
        )
        logger.info(
            "get_merchant_location_key response: status=%d body=%s",
            response.status_code,
            response.text,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("total", 0) < 1:
            logger.info("get_merchant_location_key degraded: account reports zero inventory locations")
            return None
        return data["locations"][0]["merchantLocationKey"]
    except httpx.HTTPStatusError as e:
        logger.info("get_merchant_location_key degraded: %d %s", e.response.status_code, e.response.text[:300])
        return None
    except (httpx.HTTPError, KeyError, IndexError) as e:
        logger.info("get_merchant_location_key degraded: %r", e)
        return None


# One-time setup: a single merchant location, shared by every draft this app creates.
# This is a single-seller portfolio app, so one fixed key (rather than a per-listing or
# user-chosen one) is deliberate — create_inventory_location() is only ever meant to be
# called once per eBay account, from the frontend's "Set up shipping location" form.
DEFAULT_MERCHANT_LOCATION_KEY = "agentx-default-location"


def create_inventory_location(
    config: EbayConfig,
    token: str,
    address: dict[str, str],
    name: str = "Main Location",
    location_instructions: str | None = None,
    merchant_location_key: str = DEFAULT_MERCHANT_LOCATION_KEY,
) -> None:
    payload: dict = {
        "location": {"address": address},
        "name": name,
        "merchantLocationStatus": "ENABLED",
        "locationTypes": ["WAREHOUSE"],
    }
    if location_instructions:
        payload["locationInstructions"] = location_instructions

    headers = _auth_headers(token)
    logger.info(
        "createInventoryLocation request: url=%s payload=%s",
        f"{config.api_base}/sell/inventory/v1/location/{merchant_location_key}",
        payload,
    )
    response = httpx.post(
        f"{config.api_base}/sell/inventory/v1/location/{merchant_location_key}",
        headers=headers,
        json=payload,
        timeout=20.0,
    )
    logger.info(
        "createInventoryLocation response: status=%d headers=%s body=%s",
        response.status_code,
        dict(response.headers),
        response.text,
    )
    response.raise_for_status()


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
            if items:
                policies[offer_key] = items[0][id_key]
        except httpx.HTTPStatusError as e:
            logger.info("get_listing_policies(%s) degraded: %d %s", endpoint, e.response.status_code, e.response.text[:300])
        except (httpx.HTTPError, KeyError, IndexError) as e:
            logger.info("get_listing_policies(%s) degraded: %r", endpoint, e)
    return policies


def _build_offer_payload(
    sku: str,
    price: float,
    currency: str,
    quantity: int,
    category_id: str | None,
    merchant_location_key: str | None,
    listing_policies: dict[str, str],
) -> dict:
    payload = {
        "sku": sku,
        "marketplaceId": _MARKETPLACE_ID,
        "format": "FIXED_PRICE",
        "availableQuantity": quantity,
        "pricingSummary": {"price": {"value": f"{price:.2f}", "currency": currency}},
    }
    if category_id:
        payload["categoryId"] = category_id
    if merchant_location_key:
        payload["merchantLocationKey"] = merchant_location_key
    if listing_policies:
        payload["listingPolicies"] = listing_policies
    return payload


def create_offer(
    config: EbayConfig,
    token: str,
    sku: str,
    price: float,
    currency: str,
    quantity: int,
    category_id: str | None,
    merchant_location_key: str | None,
    listing_policies: dict[str, str],
) -> str:
    payload = _build_offer_payload(sku, price, currency, quantity, category_id, merchant_location_key, listing_policies)
    logger.info("createOffer request: url=%s payload=%s", f"{config.api_base}/sell/inventory/v1/offer", payload)
    response = httpx.post(
        f"{config.api_base}/sell/inventory/v1/offer",
        headers=_auth_headers(token),
        json=payload,
        timeout=20.0,
    )
    logger.info(
        "createOffer response: status=%d headers=%s body=%s",
        response.status_code,
        dict(response.headers),
        response.text,
    )
    response.raise_for_status()
    return response.json()["offerId"]


def get_offer(config: EbayConfig, token: str, offer_id: str) -> dict:
    """Fetch a previously-created offer directly from eBay's API.

    Useful for verifying a draft actually exists without relying on eBay's
    sandbox Seller Hub web UI, which is known to be far less reliable than
    production's.
    """
    response = httpx.get(
        f"{config.api_base}/sell/inventory/v1/offer/{offer_id}",
        headers=_auth_headers(token),
        timeout=15.0,
    )
    response.raise_for_status()
    return response.json()


class DraftListingResult(BaseModel):
    sku: str
    offer_id: str
    included: list[str]
    missing: list[str]
    notes: list[str]
    aspects: dict[str, list[str]] = {}


def create_draft_listing(
    identification: ProductIdentification,
    upload_id: str,
    image_url: str,
    price: float,
    weight_lbs: float,
    currency: str = "USD",
    quantity: int = 1,
) -> DraftListingResult:
    config = load_ebay_config()
    token = get_valid_access_token(config)
    sku = generate_sku(upload_id)

    included: list[str] = []
    missing: list[str] = []
    notes: list[str] = []

    # Category must be known before the inventory item is created, since the item's
    # condition has to already be one this category allows (see resolve_condition) —
    # eBay validates condition-vs-category downstream and rejects the mismatch with a
    # cryptic errorId 25021, so this order minimizes the chance of that round-trip.
    query = build_query(identification)
    category_id = suggest_category_id(config, token, query)
    if category_id:
        included.append("category")
    else:
        missing.append("category")
        notes.append("No eBay category suggestion found — set one manually before publishing.")

    condition_enum, condition_confirmed = resolve_condition(config, token, category_id, identification.condition)
    if category_id and not condition_confirmed:
        notes.append(
            f"Couldn't confirm '{identification.condition}' is a valid eBay condition for the detected "
            f"category — used a best-effort guess ({condition_enum}); double-check it before publishing."
        )

    aspects: dict[str, list[str]] = {}
    if category_id:
        required_aspects = get_required_aspects(config, token, category_id)
        aspects, unresolved_aspects = resolve_aspects(identification, required_aspects)
        if unresolved_aspects:
            # Unlike merchant_location/listing_policies (only required to *publish*),
            # eBay validates item specifics when the offer/inventory item is created —
            # letting this through would just fail moments later with the same cryptic
            # errorId 25002 this whole function exists to avoid. Fail fast with a clear
            # message instead of spending an eBay round-trip on a guaranteed rejection.
            # Note: this can't be fixed in Seller Hub either — Seller Hub has no view
            # into an unpublished offer at all (see module docstring).
            raise RuntimeError(
                "eBay requires a value for these item specifics in the detected category, and it "
                f"couldn't be determined automatically from the photo: {', '.join(unresolved_aspects)}. "
                "Try again with a clearer photo or a more descriptive item name."
            )
    elif identification.brand:
        # No category means no aspect requirements are known at all — still send Brand
        # since we have it and it's almost always accepted regardless of category.
        aspects = {"Brand": [identification.brand]}

    create_or_replace_inventory_item(
        config, token, sku, identification, image_url, condition_enum, aspects, weight_lbs, quantity
    )
    included.append("inventory_item")

    merchant_location_key = get_merchant_location_key(config, token)
    if merchant_location_key:
        included.append("merchant_location")
    else:
        missing.append("merchant_location")
        notes.append("No shipping location set up on this eBay account — add one in Seller Hub before publishing.")

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

    offer_id = create_offer(
        config, token, sku, price, currency, quantity, category_id, merchant_location_key, listing_policies
    )
    included.append("offer")

    return DraftListingResult(
        sku=sku,
        offer_id=offer_id,
        included=included,
        missing=missing,
        notes=notes,
        aspects=aspects,
    )


def _listing_url(config: EbayConfig, listing_id: str) -> str:
    base = "https://www.sandbox.ebay.com" if config.environment == "sandbox" else "https://www.ebay.com"
    return f"{base}/itm/{listing_id}"


class PublishResult(BaseModel):
    listing_id: str
    listing_url: str


def publish_offer(config: EbayConfig, token: str, offer_id: str) -> PublishResult:
    """Publish a previously-created offer, making it a real, live, publicly
    purchasable eBay listing. Everything upstream of this call only staged a
    draft that was invisible outside eBay's raw API — this is the one
    genuinely consequential write in this codebase, only ever reached via an
    explicit user action after reviewing a full preview in our own UI.
    """
    response = httpx.post(
        f"{config.api_base}/sell/inventory/v1/offer/{offer_id}/publish/",
        headers=_auth_headers(token),
        timeout=20.0,
    )
    logger.info(
        "publishOffer response: status=%d headers=%s body=%s",
        response.status_code,
        dict(response.headers),
        response.text,
    )
    response.raise_for_status()
    listing_id = response.json()["listingId"]
    return PublishResult(listing_id=listing_id, listing_url=_listing_url(config, listing_id))
