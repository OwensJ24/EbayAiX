"""eBay Browse API: cheap, direct comparable-listing lookup (no LLM calls at all).

Uses the Client Credentials grant (an application-level token representing the
app itself, not a user) — this is a separate, simpler auth flow from the user
OAuth handshake in oauth.py/token_store.py, and needs no browser consent step.

Always uses PRODUCTION eBay credentials (EBAY_PROD_APP_ID/EBAY_PROD_CERT_ID),
even though the rest of the app defaults to sandbox: eBay's sandbox environment
has essentially no real search/catalog data — item_summary/search reliably
returns total: 0 there (a well-documented, longstanding eBay limitation, not a
bug here). This is a read-only, public-data search with no side effects, so
using production credentials carries none of the risk that using production
for listing creation would.
"""

from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel

from src.ebay.config import EbayBrowseConfig, load_ebay_browse_config

if TYPE_CHECKING:
    from src.agents.vision_subagent import ProductIdentification

_APPLICATION_SCOPE = "https://api.ebay.com/oauth/api_scope"
_MARKETPLACE_ID = "EBAY_US"

_cached_token: str | None = None
_cached_token_expires_at: float = 0.0


class EbayComp(BaseModel):
    title: str
    price: float
    currency: str
    condition: str | None = None
    item_url: str | None = None


def build_query(identification: "ProductIdentification") -> str:
    """Build a reasonable eBay search query from an item identification.

    Shared by /api/price (comp search) and the Taxonomy API category lookup in
    listing.py, so both use identical query logic.
    """
    if identification.brand and identification.model_number:
        return f"{identification.brand} {identification.model_number}"
    if identification.brand and not identification.item_name.lower().startswith(identification.brand.lower()):
        return f"{identification.brand} {identification.item_name}"
    return identification.item_name


def _basic_auth_header(config: EbayBrowseConfig) -> str:
    raw = f"{config.app_id}:{config.cert_id}".encode()
    return base64.b64encode(raw).decode()


def get_application_access_token(config: EbayBrowseConfig) -> str:
    """Client Credentials grant — cached in memory until it's close to expiring."""
    global _cached_token, _cached_token_expires_at

    if _cached_token and time.time() < _cached_token_expires_at - 60:
        return _cached_token

    response = httpx.post(
        config.token_url,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {_basic_auth_header(config)}",
        },
        data={"grant_type": "client_credentials", "scope": _APPLICATION_SCOPE},
        timeout=15.0,
    )
    response.raise_for_status()
    payload = response.json()

    _cached_token = payload["access_token"]
    _cached_token_expires_at = time.time() + payload["expires_in"]
    return _cached_token


def search_comparable_listings(query: str, limit: int = 3) -> list[EbayComp]:
    config = load_ebay_browse_config()
    token = get_application_access_token(config)

    response = httpx.get(
        f"{config.api_base}/buy/browse/v1/item_summary/search",
        headers={
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": _MARKETPLACE_ID,
        },
        params={"q": query, "limit": limit},
        timeout=15.0,
    )
    response.raise_for_status()
    data = response.json()

    comps = []
    for item in data.get("itemSummaries", [])[:limit]:
        price_info = item.get("price") or {}
        comps.append(EbayComp(
            title=item.get("title", ""),
            price=float(price_info.get("value", 0)),
            currency=price_info.get("currency", "USD"),
            condition=item.get("condition"),
            item_url=item.get("itemWebUrl"),
        ))
    return comps
