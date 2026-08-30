"""Supabase Storage client (native REST API).

Uses the Storage HTTP API ({supabase_url}/storage/v1/...) with the secret
API key instead of the S3-compatible endpoint — no boto3.
"""

import httpx
from httpx import HTTPStatusError, TransportError

from ..config import settings
from .retry import retry_async


def _base_url() -> str:
    if not settings.supabase_url:
        raise RuntimeError("configure PRODRAG_SUPABASE_URL")
    return settings.supabase_url.rstrip("/") + "/storage/v1"


def _headers() -> dict:
    if not settings.supabase_secret_key:
        raise RuntimeError("configure PRODRAG_SUPABASE_SECRET_KEY")
    return {
        "apikey": settings.supabase_secret_key,
        "Authorization": f"Bearer {settings.supabase_secret_key}",
    }


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return isinstance(exc, TransportError)


async def _request(method: str, url: str, *, content: bytes | None = None,
                   json_body: dict | None = None, extra: dict | None = None) -> httpx.Response:
    async def _call():
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.request(
                method, url, content=content, json=json_body,
                headers={**_headers(), **(extra or {})},
            )
            resp.raise_for_status()
            return resp

    return await retry_async(_call, is_transient=_is_transient, label=f"storage.{method} {url}")


async def ensure_bucket() -> None:
    buckets = (await _request("GET", f"{_base_url()}/bucket")).json()
    if any(b.get("id") == settings.storage_bucket for b in buckets):
        return
    try:
        await _request(
            "POST",
            f"{_base_url()}/bucket",
            json_body={"id": settings.storage_bucket, "name": settings.storage_bucket, "public": False},
        )
    except HTTPStatusError as exc:
        raise RuntimeError(
            f"bucket '{settings.storage_bucket}' is missing and could not be created "
            f"({exc}). Create it in Supabase Storage (dashboard or storage.buckets SQL) "
            "and retry."
        )


async def put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    url = f"{_base_url()}/object/{settings.storage_bucket}/{key}"
    await _request(
        "POST", url,
        content=data,
        extra={"Content-Type": content_type, "x-upsert": "true"},
    )


async def get_bytes(key: str) -> bytes:
    url = f"{_base_url()}/object/{settings.storage_bucket}/{key}"
    return (await _request("GET", url)).content


async def delete_object(key: str) -> None:
    url = f"{_base_url()}/object/{settings.storage_bucket}/{key}"
    try:
        await _request("DELETE", url)
    except HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise
