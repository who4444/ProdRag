"""Spec-driven API tests for /repo endpoints (M1).

Covers auth, validation, list/get/delete, and an offline end-to-end
index -> search -> delete roundtrip against the in-memory Qdrant.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("PRODRAG_API_TOKEN", "test-token")
    from app.config import settings

    monkeypatch.setattr(settings, "api_token", "test-token")

    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setattr("app.core.storage.ensure_bucket", AsyncMock())
    monkeypatch.setattr("app.core.vectorstore.ensure_collections", AsyncMock())
    monkeypatch.setattr("app.core.vectorstore.close_client", AsyncMock())

    fake_redis = AsyncMock()
    fake_redis.ping = AsyncMock(return_value=True)

    async def _fake_create_pool(*args, **kwargs):
        return fake_redis

    monkeypatch.setattr("app.main.create_pool", _fake_create_pool)
    monkeypatch.setattr("app.main.AsyncOpenAI", MagicMock(return_value=MagicMock()))

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.redis = fake_redis
        yield client, app


def _headers():
    return {"Authorization": "Bearer test-token"}


def test_repo_requires_auth(api_client):
    client, _ = api_client
    assert client.post("/repo/index", json={"path": "."}).status_code == 401
    assert client.get("/repo/list").status_code == 401


def test_repo_index_validates_request(api_client):
    client, _ = api_client
    resp = client.post("/repo/index", headers=_headers(), json={"path": ""})
    assert resp.status_code == 422


def test_repo_index_missing_path_is_400(api_client):
    client, _ = api_client
    resp = client.post("/repo/index", headers=_headers(), json={"path": "/nope/definitely/not/here"})
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"]


def test_repo_index_returns_stats(api_client, monkeypatch):
    from unittest.mock import AsyncMock

    from app.repo.schemas import RepoStats

    stats = RepoStats(repo_id="abc123", name="demo", root="/tmp/demo", files=3, chunks=9, skipped=1, runtime_ms=12)
    mock = AsyncMock(return_value=stats)
    monkeypatch.setattr("app.api.repo.index_repo", mock)

    client, _ = api_client
    resp = client.post("/repo/index", headers=_headers(), json={"path": "/tmp/demo", "name": "demo"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["repo_id"] == "abc123"
    assert body["chunks"] == 9
    assert mock.await_count == 1


def test_repo_list_and_delete(api_client, monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setattr("app.api.repo.db.repo_list", AsyncMock(return_value=[{"id": "r1"}]))
    monkeypatch.setattr("app.api.repo.db.repo_get", AsyncMock(return_value=None))
    monkeypatch.setattr("app.api.repo.db.repo_delete", AsyncMock())
    monkeypatch.setattr("app.api.repo.vectorstore.delete_code_by_repo", AsyncMock())

    client, _ = api_client
    assert client.get("/repo/list", headers=_headers()).json() == [{"id": "r1"}]
    assert client.get("/repo/nope", headers=_headers()).status_code == 404
    assert client.delete("/repo/r1", headers=_headers()).json() == {"deleted": "r1"}


def test_offline_index_search_delete_roundtrip(tmp_path, monkeypatch):
    """M1 acceptance: offline mode indexes a real tree and finds code by query."""
    from app.config import settings

    monkeypatch.setattr(settings, "offline", True)
    monkeypatch.setattr(settings, "api_token", "test-token")

    repo = tmp_path / "demo_repo"
    (repo / "svc").mkdir(parents=True)
    (repo / "svc" / "router.py").write_text(
        "def route_payments(order):\n    return order\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("# demo\n", encoding="utf-8")

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        headers = {"Authorization": "Bearer test-token"}

        indexed = client.post("/repo/index", headers=headers, json={"path": str(repo)})
        assert indexed.status_code == 200, indexed.text
        stats = indexed.json()
        assert stats["files"] == 2
        assert stats["chunks"] >= 2
        rid = stats["repo_id"]

        listed = client.get("/repo/list", headers=headers).json()
        assert any(row["id"] == rid for row in listed)

        found = client.post(
            "/repo/search",
            headers=headers,
            json={"query": "route payments order", "repo_id": rid, "k": 5},
        ).json()
        assert any(hit["path"] == "svc/router.py" for hit in found["hits"])

        assert client.delete(f"/repo/{rid}", headers=headers).json() == {"deleted": rid}
        after = client.post(
            "/repo/search", headers=headers, json={"query": "route payments", "repo_id": rid}
        ).json()
        assert after["hits"] == []
