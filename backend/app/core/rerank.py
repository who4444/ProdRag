"""Cross-encoder reranking via an optional Modal GPU service.

Mirrors the embed/clip services: no-op (None) when PRODRAG_RERANK_SERVICE_URL
is unset, so hybrid results keep their RRF order.
"""

import httpx
from httpx import HTTPStatusError, TransportError

from ..config import settings
from .retry import retry_async


def enabled() -> bool:
    return bool((settings.rerank_service_url or "").strip())


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, TransportError)


async def rerank(query: str, texts: list[str]) -> list[float] | None:
    """Score each (query, text) pair with the cross-encoder; None when not configured."""
    if not enabled() or not texts:
        return None
    headers = {}
    if settings.rerank_service_token:
        headers["Authorization"] = f"Bearer {settings.rerank_service_token}"

    async def _call():
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                settings.rerank_service_url.rstrip("/"),
                json={"query": query, "texts": texts},
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()["scores"]

    return await retry_async(_call, is_transient=_is_transient, label="rerank_service")