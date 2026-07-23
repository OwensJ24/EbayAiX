"""eBay OAuth 2.0 authorization-code flow for user access tokens."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import httpx

from src.ebay.config import DEFAULT_SCOPES, EbayConfig


@dataclass(frozen=True)
class EbayTokens:
    access_token: str
    refresh_token: str
    access_token_expires_at: float
    refresh_token_expires_at: float

    @property
    def is_access_token_expired(self) -> bool:
        return time.time() >= self.access_token_expires_at - 60


def build_authorize_url(
    config: EbayConfig,
    state: str | None = None,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
) -> str:
    params = {
        "client_id": config.app_id,
        "redirect_uri": config.ru_name,
        "response_type": "code",
        "scope": " ".join(scopes),
    }
    if state:
        params["state"] = state
    return str(httpx.URL(config.authorize_url, params=params))


def _basic_auth_header(config: EbayConfig) -> str:
    raw = f"{config.app_id}:{config.cert_id}".encode()
    return base64.b64encode(raw).decode()


def exchange_code_for_tokens(config: EbayConfig, code: str) -> EbayTokens:
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {_basic_auth_header(config)}",
    }
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": config.ru_name,
    }
    response = httpx.post(config.token_url, headers=headers, data=data, timeout=15.0)
    response.raise_for_status()
    payload = response.json()
    now = time.time()
    return EbayTokens(
        access_token=payload["access_token"],
        refresh_token=payload["refresh_token"],
        access_token_expires_at=now + payload["expires_in"],
        refresh_token_expires_at=now + payload["refresh_token_expires_in"],
    )


def refresh_access_token(
    config: EbayConfig,
    tokens: EbayTokens,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
) -> EbayTokens:
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {_basic_auth_header(config)}",
    }
    data = {
        "grant_type": "refresh_token",
        "refresh_token": tokens.refresh_token,
        "scope": " ".join(scopes),
    }
    response = httpx.post(config.token_url, headers=headers, data=data, timeout=15.0)
    response.raise_for_status()
    payload = response.json()
    now = time.time()
    # eBay does not rotate the refresh token on a refresh grant — keep the original.
    return EbayTokens(
        access_token=payload["access_token"],
        refresh_token=tokens.refresh_token,
        access_token_expires_at=now + payload["expires_in"],
        refresh_token_expires_at=tokens.refresh_token_expires_at,
    )
