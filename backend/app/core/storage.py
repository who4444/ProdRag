"""S3 object storage backed by Supabase Storage (S3-compatible API).

Supabase's endpoint carries a path (https://<ref>.supabase.co/storage/v1/s3),
so we use boto3 (the minio client rejects path endpoints). The scheme comes
from the endpoint URL itself.
"""

import asyncio
import io
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError

from ..config import settings


def _s3_endpoint() -> str:
    if settings.s3_endpoint:
        return settings.s3_endpoint.rstrip("/")
    if settings.supabase_url:
        return settings.supabase_url.rstrip("/") + "/storage/v1/s3"
    raise RuntimeError("configure PRODRAG_SUPABASE_URL or PRODRAG_S3_ENDPOINT")


@lru_cache(maxsize=1)
def _client():
    return boto3.client(
        "s3",
        endpoint_url=_s3_endpoint(),
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
    )


async def ensure_bucket() -> None:
    client = _client()
    try:
        await asyncio.to_thread(client.head_bucket, Bucket=settings.storage_bucket)
    except ClientError:
        try:
            await asyncio.to_thread(client.create_bucket, Bucket=settings.storage_bucket)
        except Exception as exc:
            raise RuntimeError(
                f"bucket '{settings.storage_bucket}' is missing and could not be created "
                f"({exc}). Create it in Supabase Storage (dashboard or storage.buckets SQL) "
                "and retry."
            )


async def put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    client = _client()
    await asyncio.to_thread(
        client.put_object,
        Bucket=settings.storage_bucket,
        Key=key,
        Body=io.BytesIO(data),
        ContentType=content_type,
    )


async def get_bytes(key: str) -> bytes:
    client = _client()
    resp = await asyncio.to_thread(
        client.get_object, Bucket=settings.storage_bucket, Key=key
    )
    return resp["Body"].read()
