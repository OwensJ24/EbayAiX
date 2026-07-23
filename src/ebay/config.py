"""eBay Sell API configuration, loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

_SANDBOX_API_BASE = "https://api.sandbox.ebay.com"
_SANDBOX_AUTH_BASE = "https://auth.sandbox.ebay.com"
_PRODUCTION_API_BASE = "https://api.ebay.com"
_PRODUCTION_AUTH_BASE = "https://auth.ebay.com"

# Scope needed to create/manage draft listings via the Sell Inventory API.
SELL_INVENTORY_SCOPE = "https://api.ebay.com/oauth/api_scope/sell.inventory"
# Read-only scope for fetching existing business policies/merchant location, so a
# draft listing can include them when the account already has them set up.
SELL_ACCOUNT_READONLY_SCOPE = "https://api.ebay.com/oauth/api_scope/sell.account.readonly"
DEFAULT_SCOPES: tuple[str, ...] = (SELL_INVENTORY_SCOPE, SELL_ACCOUNT_READONLY_SCOPE)

_REQUIRED_ENV_VARS = ("EBAY_APP_ID", "EBAY_CERT_ID", "EBAY_RU_NAME")


@dataclass(frozen=True)
class EbayConfig:
    app_id: str
    cert_id: str
    ru_name: str
    environment: str  # "sandbox" or "production"

    @property
    def api_base(self) -> str:
        return _SANDBOX_API_BASE if self.environment == "sandbox" else _PRODUCTION_API_BASE

    @property
    def auth_base(self) -> str:
        return _SANDBOX_AUTH_BASE if self.environment == "sandbox" else _PRODUCTION_AUTH_BASE

    @property
    def token_url(self) -> str:
        return f"{self.api_base}/identity/v1/oauth2/token"

    @property
    def authorize_url(self) -> str:
        return f"{self.auth_base}/oauth2/authorize"


def load_ebay_config() -> EbayConfig:
    environment = os.environ.get("EBAY_ENVIRONMENT", "sandbox").strip().lower()
    if environment not in {"sandbox", "production"}:
        raise ValueError(f"EBAY_ENVIRONMENT must be 'sandbox' or 'production', got {environment!r}")

    missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing required eBay env vars: {', '.join(missing)} — add them to .env (see .env.example)"
        )

    return EbayConfig(
        app_id=os.environ["EBAY_APP_ID"],
        cert_id=os.environ["EBAY_CERT_ID"],
        ru_name=os.environ["EBAY_RU_NAME"],
        environment=environment,
    )


_REQUIRED_BROWSE_ENV_VARS = ("EBAY_PROD_APP_ID", "EBAY_PROD_CERT_ID")


@dataclass(frozen=True)
class EbayBrowseConfig:
    """Always production — eBay's sandbox has no real search/catalog data to query."""

    app_id: str
    cert_id: str

    @property
    def api_base(self) -> str:
        return _PRODUCTION_API_BASE

    @property
    def token_url(self) -> str:
        return f"{self.api_base}/identity/v1/oauth2/token"


def load_ebay_browse_config() -> EbayBrowseConfig:
    missing = [name for name in _REQUIRED_BROWSE_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing required eBay env vars: {', '.join(missing)} — add them to .env (see .env.example). "
            "These must be your PRODUCTION keyset — eBay's sandbox returns no real search results."
        )

    return EbayBrowseConfig(
        app_id=os.environ["EBAY_PROD_APP_ID"],
        cert_id=os.environ["EBAY_PROD_CERT_ID"],
    )
