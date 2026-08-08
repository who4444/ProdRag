import hashlib
import json
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .config import settings

_client: AsyncQdrantClient | None = None


async def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def ensure_collections() -> None:
    client = await get_client()
    for name, dim in (
        (settings.collection_text, settings.embedding_dim),
        (settings.collection_image, settings.vision_dim),
    ):
        if await client.collection_exists(name):
            continue
        try:
            await client.create_collection(
                name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
        except UnexpectedResponse as exc:
            if exc.status_code != 409:
                raise


def _point_id(kind: str, payload: dict) -> str:
    """Deterministic id from payload -> idempotent re-ingestion (upsert overwrites)."""
    raw = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256((kind + raw).encode()).digest()[:16]
    return str(uuid.UUID(int=int.from_bytes(digest, "big")))


async def upsert_text(items: list[tuple[dict, list[float]]]) -> int:
    client = await get_client()
    points = [PointStruct(id=_point_id("text", p), vector=v, payload=p) for p, v in items]
    if not points:
        return 0
    await client.upsert(settings.collection_text, points)
    return len(points)


async def upsert_image(items: list[tuple[dict, list[float]]]) -> int:
    client = await get_client()
    points = [PointStruct(id=_point_id("image", p), vector=v, payload=p) for p, v in items]
    if not points:
        return 0
    await client.upsert(settings.collection_image, points)
    return len(points)


async def search_text(vector: list[float], k: int, source: str | None = None) -> list[dict]:
    client = await get_client()
    query_filter = None
    if source:
        query_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        )
    res = await client.query_points(
        settings.collection_text,
        query=vector,
        limit=k,
        query_filter=query_filter,
        with_payload=True,
    )
    return [{"score": h.score, **h.payload} for h in res.points]


async def search_images(vector: list[float], k: int) -> list[dict]:
    client = await get_client()
    res = await client.query_points(
        settings.collection_image,
        query=vector,
        limit=k,
        with_payload=True,
    )
    return [{"score": h.score, **h.payload} for h in res.points]
