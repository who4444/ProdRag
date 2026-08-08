"""Text embeddings. Remote-first: calls the Modal GPU service
(deploy/modal/embed_service.py, BAAI/bge-m3) when PRODRAG_EMBED_SERVICE_URL
is set. Falls back to OpenAI text-embedding-3-small for local dev.
"""

from functools import lru_cache

import httpx
from openai import AsyncOpenAI

from ..config import settings


@lru_cache(maxsize=1)
def get_openai() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


def _remote_url() -> str | None:
    url = (settings.embed_service_url or "").strip().rstrip("/")
    return url or None


async def _remote_embed(texts: list[str], query: bool) -> list[list[float]]:
    headers = {}
    if settings.embed_service_token:
        headers["Authorization"] = f"Bearer {settings.embed_service_token}"
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            _remote_url(),
            json={"data": texts, "query": query},
            headers=headers,
        )
        resp.raise_for_status()
    payload = resp.json()
    if payload["dim"] != settings.embedding_dim:
        raise RuntimeError(
            f"embed service dim {payload['dim']} != PRODRAG_EMBEDDING_DIM "
            f"{settings.embedding_dim} — recreate the text collection"
        )
    return payload["embeddings"]


async def embed_texts(texts: list[str], query: bool = False) -> list[list[float]]:
    if not texts:
        return []
    if _remote_url():
        return await _remote_embed(texts, query)
    resp = await get_openai().embeddings.create(
        model=settings.embedding_model, input=texts
    )
    return [d.embedding for d in resp.data]
