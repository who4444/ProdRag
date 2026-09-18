"""Text embeddings. Remote-first: calls the Modal GPU service
(deploy/modal/embed_service.py, BAAI/bge-m3) when PRODRAG_EMBED_SERVICE_URL
is set. Falls back to OpenAI text-embedding-3-small for local dev, or to a
deterministic lexical hash embedder in offline mode.
"""

import hashlib
import math
from functools import lru_cache

import httpx
from httpx import HTTPStatusError, TransportError
from openai import AsyncOpenAI

from ...config import settings
from ..retry import retry_async
from ..sparse import tokens as _tokens


def _hash_embed(text: str, dim: int) -> list[float]:
    """Deterministic lexical embedding (offline only) — no model, no network."""
    vec = [0.0] * dim
    for tok in _tokens(text):
        idx = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


@lru_cache(maxsize=1)
def get_openai() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


def _remote_url() -> str | None:
    url = (settings.embed_service_url or "").strip().rstrip("/")
    return url or None


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, TransportError)


async def _remote_embed(texts: list[str], query: bool) -> list[list[float]]:
    headers = {}
    if settings.embed_service_token:
        headers["Authorization"] = f"Bearer {settings.embed_service_token}"

    async def _call():
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                _remote_url(),
                json={"data": texts, "query": query},
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    payload = await retry_async(
        _call, is_transient=_is_transient, label="embed_service"
    )
    if payload["dim"] != settings.embedding_dim:
        raise RuntimeError(
            f"embed service dim {payload['dim']} != PRODRAG_EMBEDDING_DIM "
            f"{settings.embedding_dim} — recreate the text collection"
        )
    return payload["embeddings"]


async def embed_texts(texts: list[str], query: bool = False) -> list[list[float]]:
    if not texts:
        return []
    if settings.offline:
        return [_hash_embed(t, settings.embedding_dim) for t in texts]
    if _remote_url():
        return await _remote_embed(texts, query)
    resp = await get_openai().embeddings.create(
        model=settings.embedding_model, input=texts
    )
    return [d.embedding for d in resp.data]
