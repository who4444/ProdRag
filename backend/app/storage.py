import asyncio
import io
from functools import lru_cache

from minio import Minio

from .config import settings


@lru_cache(maxsize=1)
def _client() -> Minio:
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


async def ensure_bucket() -> None:
    client = _client()
    exists = await asyncio.to_thread(client.bucket_exists, settings.minio_bucket)
    if not exists:
        await asyncio.to_thread(client.make_bucket, settings.minio_bucket)


async def put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    client = _client()
    await asyncio.to_thread(
        client.put_object,
        settings.minio_bucket,
        key,
        io.BytesIO(data),
        len(data),
        content_type=content_type,
    )


async def get_bytes(key: str) -> bytes:
    client = _client()
    resp = await asyncio.to_thread(client.get_object, settings.minio_bucket, key)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()
