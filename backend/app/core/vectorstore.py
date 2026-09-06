import hashlib
import json
import logging
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Modifier,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from ..config import settings
from .sparse import to_sparse

logger = logging.getLogger(__name__)

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
        (settings.memory_collection, settings.embedding_dim),
    ):
        if await client.collection_exists(name):
            if name == settings.collection_text and not await _has_sparse(name, client):
                logger.warning(
                    "text collection '%s' lacks the 'bm25' sparse vector — delete it "
                    "and re-ingest to enable hybrid retrieval",
                    name,
                )
            continue
        try:
            if name == settings.collection_text:
                await client.create_collection(
                    name,
                    vectors_config={"": VectorParams(size=dim, distance=Distance.COSINE)},
                    sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
                )
            else:
                await client.create_collection(
                    name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                )
        except UnexpectedResponse as exc:
            if exc.status_code != 409:
                raise


async def _has_sparse(name: str, client) -> bool:
    info = await client.get_collection(name)
    sparse = info.config.params.sparse_vectors
    return bool(sparse and "bm25" in sparse)


def _point_id(kind: str, payload: dict) -> str:
    """Deterministic id from payload -> idempotent re-ingestion (upsert overwrites)."""
    raw = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256((kind + raw).encode()).digest()[:16]
    return str(uuid.UUID(int=int.from_bytes(digest, "big")))


async def point_id(kind: str, payload: dict) -> str:
    return _point_id(kind, payload)


async def upsert_text(items: list[tuple[dict, list[float]]]) -> int:
    client = await get_client()
    points = [
        PointStruct(
            id=_point_id("text", p),
            vector={"": v, "bm25": SparseVector(**to_sparse(p["text"]))},
            payload=p,
        )
        for p, v in items
    ]
    if not points:
        return 0
    try:
        await client.upsert(settings.collection_text, points)
    except UnexpectedResponse as exc:
        # Old collection without bm25 (see ensure_collections) — fall back to dense only
        if "bm25" in str(exc).lower() or "vector name" in str(exc).lower():
            logger.warning("upsert_text bm25 failed on '%s' — falling back to dense (delete collection to enable hybrid)", settings.collection_text)
            dense_points = [PointStruct(id=pid.id, vector={"": vec}, payload=payload) for (payload, vec), pid in zip(items, points)]
            await client.upsert(settings.collection_text, dense_points)
        else:
            raise
    return len(points)


async def upsert_image(items: list[tuple[dict, list[float]]]) -> int:
    client = await get_client()
    points = [PointStruct(id=_point_id("image", p), vector=v, payload=p) for p, v in items]
    if not points:
        return 0
    await client.upsert(settings.collection_image, points)
    return len(points)


def _hybrid_prefetch(dense: list[float], sparse: dict, qf: Filter | None, prefetch_k: int) -> list[Prefetch]:
    """Prefetch list for dense + sparse (BM25) retrieval fused with RRF."""
    return [
        Prefetch(query=dense, limit=prefetch_k, filter=qf),
        Prefetch(query=SparseVector(**sparse), using="bm25", limit=prefetch_k, filter=qf),
    ]


async def search_text(
    vector: list[float],
    k: int,
    source: str | None = None,
    sparse_query: dict | None = None,
    prefetch_k: int | None = None,
) -> list[dict]:
    client = await get_client()
    qf = None
    if source:
        qf = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
    if sparse_query and prefetch_k:
        try:
            res = await client.query_points(
                settings.collection_text,
                prefetch=_hybrid_prefetch(vector, sparse_query, qf, prefetch_k),
                query=FusionQuery(fusion=Fusion.RRF),
                limit=k,
                with_payload=True,
            )
            return [{"score": h.score, **h.payload} for h in res.points]
        except UnexpectedResponse:
            # Old collection without the 'bm25' sparse vector (see ensure_collections).
            logger.warning("hybrid search failed on '%s' — falling back to dense", settings.collection_text)
    res = await client.query_points(
        settings.collection_text,
        query=vector,
        limit=k,
        query_filter=qf,
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


async def delete_points(point_ids: list[str]) -> None:
    client = await get_client()
    if not point_ids:
        return
    await client.delete(settings.collection_text, point_ids)
    await client.delete(settings.collection_image, point_ids)


async def upsert_episodes(items: list[tuple[dict, list[float]]]) -> str | None:
    client = await get_client()
    points = [PointStruct(id=_point_id("episode", p), vector=v, payload=p) for p, v in items]
    if not points:
        return None
    await client.upsert(settings.memory_collection, points)
    return str(points[0].id)


async def search_episodes(vector: list[float], k: int) -> list[dict]:
    client = await get_client()
    res = await client.query_points(
        settings.memory_collection,
        query=vector,
        limit=k,
        with_payload=True,
    )
    return [{"score": h.score, **h.payload} for h in res.points]
