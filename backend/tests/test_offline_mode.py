"""Offline mode: boot + embeddings without any external service (M0)."""

import asyncio

import pytest


def test_hash_embed_is_deterministic_normalized_and_sized(monkeypatch):
    from app.config import settings
    from app.core.embeddings.text import embed_texts

    monkeypatch.setattr(settings, "offline", True)
    a = asyncio.run(embed_texts(["sandbox subprocess runner"]))[0]
    b = asyncio.run(embed_texts(["sandbox subprocess runner"]))[0]
    c = asyncio.run(embed_texts(["completely different words"]))[0]

    assert len(a) == settings.embedding_dim
    assert a == b
    assert a != c
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6


def test_app_boots_and_health_reports_offline(monkeypatch):
    from fastapi.testclient import TestClient

    from app.config import settings

    monkeypatch.setattr(settings, "offline", True)
    monkeypatch.setattr(settings, "api_token", "offline-token")

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "offline": True}

        # auth still enforced in offline mode
        assert client.get("/rd/runs").status_code == 401


@pytest.mark.asyncio
async def test_offline_repo_search_roundtrip(monkeypatch):
    """In-memory Qdrant + hash embeddings: upsert then search finds the chunk."""
    from app.config import settings
    from app.core import vectorstore
    from app.core.embeddings.text import embed_texts
    from app.core.sparse import to_sparse

    monkeypatch.setattr(settings, "offline", True)
    await vectorstore.close_client()
    vectorstore._client = None
    await vectorstore.ensure_collections()

    payload = {
        "repo_id": "repo-1",
        "path": "backend/app/code/sandbox.py",
        "language": "python",
        "symbol": "run_sandbox",
        "kind": "function",
        "start_line": 194,
        "end_line": 236,
        "imports": ["asyncio"],
        "text": "async def run_sandbox(files): tempfile subprocess pytest demo",
    }
    vecs = await embed_texts([payload["text"]])
    n = await vectorstore.upsert_code([(payload, vecs[0])])
    assert n == 1

    q = (await embed_texts(["where is the sandbox subprocess run"], query=True))[0]
    hits = await vectorstore.search_code(q, k=3, repo_id="repo-1", sparse_query=to_sparse("sandbox subprocess"), prefetch_k=12)
    assert hits, "expected at least one code hit"
    assert hits[0]["path"] == "backend/app/code/sandbox.py"

    other = await vectorstore.search_code(q, k=3, repo_id="nope", sparse_query=to_sparse("sandbox"), prefetch_k=12)
    assert other == []

    await vectorstore.delete_code_by_repo("repo-1")
    after = await vectorstore.search_code(q, k=3, repo_id="repo-1", sparse_query=to_sparse("sandbox"), prefetch_k=12)
    assert after == []
    await vectorstore.close_client()
    vectorstore._client = None
