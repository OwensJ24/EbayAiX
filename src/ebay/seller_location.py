"""Local persistence for the seller's ship-from location (single-user, file-backed).

Trading API's AddFixedPriceItem needs only flat Item.Country + Item.PostalCode fields —
unlike the REST Inventory API's merchantLocationKey system, there's no eBay-side
location object to register or look up at all. Persisted locally instead, same style as
token_store.py's data/ebay_tokens.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOCATION_FILE = _PROJECT_ROOT / "data" / "seller_location.json"


class SellerLocation(BaseModel):
    country: str
    postal_code: str


def save_seller_location(country: str, postal_code: str) -> None:
    LOCATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCATION_FILE.write_text(json.dumps({"country": country, "postal_code": postal_code}))


def load_seller_location() -> SellerLocation | None:
    if not LOCATION_FILE.exists():
        return None
    return SellerLocation(**json.loads(LOCATION_FILE.read_text()))
