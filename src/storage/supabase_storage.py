"""Supabase Storage: hosts uploaded item photos publicly so eBay can fetch a listing's
imageUrls independent of whether this app's own Render instance is awake or even
running — Render's free tier spins down after inactivity, and eBay fetches imageUrls
lazily rather than synchronously at inventory-item-creation time (see
src/ebay/listing.py), so relying on this app's own server to keep serving the image
indefinitely was a real reliability gap, not just a resource one. Raw httpx calls to
Supabase's Storage REST API, matching this project's eBay integration style — no SDK
dependency.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupabaseStorageConfig:
    url: str
    # Supabase's newer "secret" key (sb_secret_..., Settings -> API Keys -> Secret keys)
    # — the forward-looking replacement for the legacy service_role key, used exactly
    # the same way at the HTTP level (same apikey/Authorization headers, same full
    # access via the service_role Postgres role, same BYPASSRLS behavior). Legacy
    # service_role keys still work and are slated for deprecation by end of 2026;
    # secret keys are preferred going forward and add per-key rotation plus a
    # same-origin/browser-use block that service_role keys don't have.
    secret_key: str
    bucket: str


def load_supabase_storage_config() -> SupabaseStorageConfig:
    missing = [
        name for name in ("SUPABASE_URL", "SUPABASE_SECRET_KEY", "SUPABASE_BUCKET") if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(
            f"Missing required Supabase env vars: {', '.join(missing)} — add them to .env (see .env.example)"
        )
    return SupabaseStorageConfig(
        url=os.environ["SUPABASE_URL"].rstrip("/"),
        secret_key=os.environ["SUPABASE_SECRET_KEY"],
        bucket=os.environ["SUPABASE_BUCKET"],
    )


def _object_path(upload_id: str) -> str:
    # Every upload is normalized to JPEG at save time (see app.py's
    # _validate_and_normalize_image), so this is always deterministic — nothing
    # downstream ever needs to track or look up the original file extension.
    return f"{upload_id}.jpg"


def public_url(config: SupabaseStorageConfig, upload_id: str) -> str:
    """Deterministic — never checks whether the object actually exists. Assumes
    upload_image() already succeeded earlier in the flow for this upload_id."""
    return f"{config.url}/storage/v1/object/public/{config.bucket}/{_object_path(upload_id)}"


def _auth_headers(config: SupabaseStorageConfig) -> dict[str, str]:
    return {
        "apikey": config.secret_key,
        "Authorization": f"Bearer {config.secret_key}",
    }


def upload_image(config: SupabaseStorageConfig, upload_id: str, contents: bytes) -> str:
    """Uploads image bytes, returns the public URL. The bucket must be configured as
    public (Supabase dashboard -> Storage -> the bucket -> toggle 'Public bucket') —
    this app has no way to set that via API; it's a one-time manual setup step.
    """
    path = _object_path(upload_id)
    headers = {
        **_auth_headers(config),
        "Content-Type": "image/jpeg",
        "x-upsert": "true",  # re-running identify for the same upload_id overwrites cleanly
    }
    response = httpx.post(
        f"{config.url}/storage/v1/object/{config.bucket}/{path}",
        headers=headers,
        content=contents,
        timeout=30.0,
    )
    response.raise_for_status()
    return public_url(config, upload_id)


def delete_image(config: SupabaseStorageConfig, upload_id: str) -> None:
    """Best-effort cleanup once a listing is published — failures are logged, not
    raised, since a lingering Supabase object is a much smaller problem than surfacing
    an error to the user after publishing has already succeeded.
    """
    try:
        response = httpx.request(
            "DELETE",
            f"{config.url}/storage/v1/object/{config.bucket}",
            headers={**_auth_headers(config), "Content-Type": "application/json"},
            json={"prefixes": [_object_path(upload_id)]},
            timeout=15.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.info("delete_image degraded: %r", e)
