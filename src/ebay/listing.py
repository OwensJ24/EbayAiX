"""eBay Inventory + Offer APIs: create a DRAFT listing (never published).

Flow: createOrReplaceInventoryItem -> best-effort category/location/policy
enrichment -> createOffer. Deliberately stops there — publishOffer is never
called. This boundary is this project's Human-in-the-Loop safety gate for
eBay writes: the draft sits in the seller's account for a human to review and
manually publish from eBay's own Seller Hub.
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from src.agents.vision_subagent import ProductIdentification
from src.ebay.browse import build_query
from src.ebay.config import EbayConfig, load_ebay_config
from src.ebay.token_store import get_valid_access_token

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


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def generate_sku(upload_id: str) -> str:
    return f"agentx-{upload_id}"


def _build_description(identification: ProductIdentification) -> str:
    parts = [identification.condition_notes]
    if identification.distinguishing_features:
        parts.append("Notable features:")
        parts.extend(f"- {f}" for f in identification.distinguishing_features)
    description = "\n".join(parts)
    return description[:4000]


def _build_inventory_item_payload(identification: ProductIdentification, image_url: str, quantity: int) -> dict:
    payload = {
        "condition": _CONDITION_MAP[identification.condition],
        "product": {
            "title": identification.item_name[:80],
            "description": _build_description(identification),
            "imageUrls": [image_url],
        },
        "availability": {"shipToLocationAvailability": {"quantity": quantity}},
    }
    if identification.brand:
        payload["product"]["aspects"] = {"Brand": [identification.brand]}
    return payload


def create_or_replace_inventory_item(
    config: EbayConfig, token: str, sku: str, identification: ProductIdentification, image_url: str, quantity: int = 1
) -> None:
    payload = _build_inventory_item_payload(identification, image_url, quantity)
    response = httpx.put(
        f"{config.api_base}/sell/inventory/v1/inventory_item/{sku}",
        # Content-Language is the header eBay's own docs require here. Accept-Language
        # is added too: eBay's "Invalid value for header Content-Language" (error
        # 25709) is a well-documented, cross-platform (Node.js/C#/Power Automate) case
        # of a misleading error message — multiple independent reports found sending
        # both headers resolves it, including cases where the real cause clearly
        # wasn't the header value itself (still failed even with it removed entirely).
        headers={**_auth_headers(token), "Content-Language": "en-US", "Accept-Language": "en-US"},
        json=payload,
        timeout=20.0,
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
            return None
        return suggestions[0]["category"]["categoryId"]
    except (httpx.HTTPError, KeyError, IndexError):
        return None


def get_merchant_location_key(config: EbayConfig, token: str) -> str | None:
    try:
        response = httpx.get(
            f"{config.api_base}/sell/inventory/v1/location",
            headers=_auth_headers(token),
            params={"limit": 1},
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("total", 0) < 1:
            return None
        return data["locations"][0]["merchantLocationKey"]
    except (httpx.HTTPError, KeyError, IndexError):
        return None


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
        except (httpx.HTTPError, KeyError, IndexError):
            continue
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
    response = httpx.post(
        f"{config.api_base}/sell/inventory/v1/offer",
        headers=_auth_headers(token),
        json=payload,
        timeout=20.0,
    )
    response.raise_for_status()
    return response.json()["offerId"]


def _seller_hub_url(config: EbayConfig) -> str:
    base = "https://www.sandbox.ebay.com" if config.environment == "sandbox" else "https://www.ebay.com"
    return f"{base}/sh/lst/drafts"


class DraftListingResult(BaseModel):
    sku: str
    offer_id: str
    seller_hub_url: str
    included: list[str]
    missing: list[str]
    notes: list[str]


def create_draft_listing(
    identification: ProductIdentification,
    upload_id: str,
    image_url: str,
    price: float,
    currency: str = "USD",
    quantity: int = 1,
) -> DraftListingResult:
    config = load_ebay_config()
    token = get_valid_access_token(config)
    sku = generate_sku(upload_id)

    create_or_replace_inventory_item(config, token, sku, identification, image_url, quantity)
    included = ["inventory_item"]
    missing: list[str] = []
    notes: list[str] = []

    query = build_query(identification)
    category_id = suggest_category_id(config, token, query)
    if category_id:
        included.append("category")
    else:
        missing.append("category")
        notes.append("No eBay category suggestion found — set one manually before publishing.")

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
        seller_hub_url=_seller_hub_url(config),
        included=included,
        missing=missing,
        notes=notes,
    )
