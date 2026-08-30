import httpx
import pytest

from app.core import storage
from app.config import settings


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(settings, "supabase_url", "https://example.supabase.co")
    monkeypatch.setattr(settings, "supabase_secret_key", "test-role")
    monkeypatch.setattr(settings, "storage_bucket", "prodrag-assets")


def _patch_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    class _FakeAsyncClient:
        def __init__(self, **kwargs):
            self._client = real(transport=transport)

        async def __aenter__(self):
            return self._client

        async def __aexit__(self, *exc):
            await self._client.aclose()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)


def test_put_bytes_uses_upsert_endpoint(env, monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = request.content
        return httpx.Response(200, json={"Key": "ok"})

    _patch_client(monkeypatch, handler)

    import asyncio
    asyncio.run(storage.put_bytes("docs/x/fig.png", b"PNGDATA", "image/png"))

    assert seen["url"] == "https://example.supabase.co/storage/v1/object/prodrag-assets/docs/x/fig.png"
    assert seen["headers"]["x-upsert"] == "true"
    assert seen["headers"]["content-type"] == "image/png"
    assert seen["body"] == b"PNGDATA"


def test_get_bytes_returns_content(env, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"PNGDATA")

    _patch_client(monkeypatch, handler)

    import asyncio
    data = asyncio.run(storage.get_bytes("docs/x/fig.png"))
    assert data == b"PNGDATA"


def test_transient_500_retried(env, monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, content=b"PNGDATA")

    _patch_client(monkeypatch, handler)

    import asyncio
    data = asyncio.run(storage.get_bytes("docs/x/fig.png"))
    assert data == b"PNGDATA"
    assert calls["n"] == 3