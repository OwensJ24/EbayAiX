"""Local persistence for eBay OAuth tokens (single-user, file-backed)."""

from __future__ import annotations

import json
from pathlib import Path

from src.ebay.config import EbayConfig
from src.ebay.oauth import EbayTokens, refresh_access_token

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TOKEN_FILE = _PROJECT_ROOT / "data" / "ebay_tokens.json"


def save_tokens(tokens: EbayTokens) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps({
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "access_token_expires_at": tokens.access_token_expires_at,
        "refresh_token_expires_at": tokens.refresh_token_expires_at,
    }))


def load_tokens() -> EbayTokens | None:
    if not TOKEN_FILE.exists():
        return None
    return EbayTokens(**json.loads(TOKEN_FILE.read_text()))


def clear_tokens() -> None:
    TOKEN_FILE.unlink(missing_ok=True)


def get_valid_access_token(config: EbayConfig) -> str:
    """Return a live access token, transparently refreshing it if expired."""
    tokens = load_tokens()
    if tokens is None:
        raise RuntimeError("No eBay tokens stored yet — connect your eBay account first via /ebay/connect")
    if tokens.is_access_token_expired:
        tokens = refresh_access_token(config, tokens)
        save_tokens(tokens)
    return tokens.access_token
