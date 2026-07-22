"""eBay OAuth 2.0 connect flow: authorize redirect + callback token exchange."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from src.ebay.config import load_ebay_config
from src.ebay.oauth import build_authorize_url, exchange_code_for_tokens
from src.ebay.token_store import load_tokens, save_tokens

router = APIRouter(prefix="/ebay", tags=["ebay"])

# In-memory CSRF state for the OAuth handshake — fine for a single-user local app.
_pending_state: str | None = None


@router.get("/connect")
def connect() -> RedirectResponse:
    global _pending_state
    config = load_ebay_config()
    _pending_state = secrets.token_urlsafe(16)
    return RedirectResponse(build_authorize_url(config, state=_pending_state))


@router.get("/callback")
def callback(request: Request) -> RedirectResponse:
    global _pending_state
    error = request.query_params.get("error")
    code = request.query_params.get("code")
    state = request.query_params.get("state")

    if error:
        raise HTTPException(status_code=400, detail=f"eBay authorization failed: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code from eBay")
    if state != _pending_state:
        raise HTTPException(status_code=400, detail="OAuth state mismatch — possible CSRF, please reconnect")
    _pending_state = None

    config = load_ebay_config()
    tokens = exchange_code_for_tokens(config, code)
    save_tokens(tokens)

    return RedirectResponse("/?ebay_connected=1")


@router.get("/status")
def status() -> dict:
    tokens = load_tokens()
    if tokens is None:
        return {"connected": False}
    return {"connected": True, "access_token_expired": tokens.is_access_token_expired}
