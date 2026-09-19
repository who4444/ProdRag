"""Spec-driven API tests for /runs — auth, 400/409, NDJSON, offline e2e (M3)."""

import json

import pytest
from fastapi.testclient import TestClient

from test_runs_graph_spec import RoutingClient


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


def test_runs_require_auth(api_client):
    client, _ = api_client
    assert client.get("/runs").status_code == 401
    assert client.post("/runs", json={"idea": "x", "repo_id": "r"}).status_code == 401


def test_runs_validate_request(api_client):
    client, _ = api_client
    assert client.post("/runs", headers=_headers(), json={"idea": "x"}).status_code == 422
    assert client.post("/runs", headers=_headers(), json={"idea": "", "repo_id": "r"}).status_code == 422


def test_runs_unknown_repo_is_400(api_client, monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setattr("app.feasibility.pipeline.db.repo_get", AsyncMock(return_value=None))
    client, _ = api_client
    resp = client.post("/runs", headers=_headers(), json={"idea": "add caching", "repo_id": "missing"})
    assert resp.status_code == 400
    assert "not indexed" in resp.json()["detail"]


def test_offline_run_e2e_clarify_resume_and_result(tmp_path, monkeypatch):
    """M3 acceptance: offline graph run pauses for input, resumes, persists."""
    from app.config import settings

    monkeypatch.setattr(settings, "offline", True)
    monkeypatch.setattr(settings, "api_token", "test-token")

    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "core.py").write_text("class Engine:\n    pass\n\n\ndef run():\n    pass\n", encoding="utf-8")

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        headers = {"Authorization": "Bearer test-token"}
        indexed = client.post("/repo/index", headers=headers, json={"path": str(repo)})
        assert indexed.status_code == 200, indexed.text
        repo_id = indexed.json()["repo_id"]

        app.state.client = RoutingClient(ask_clarify=True)

        with client.stream(
            "POST", "/runs", headers=headers, json={"idea": "add caching to the engine", "repo_id": repo_id}
        ) as resp:
            assert resp.status_code == 200
            assert "x-ndjson" in resp.headers.get("content-type", "")
            first = [json.loads(line) for line in resp.iter_lines() if line]

        interrupt = next(e for e in first if e.get("type") == "run" and e.get("event") == "interrupt")
        assert interrupt["kind"] == "clarify"
        run_id = interrupt["run_id"]

        app.state.client = RoutingClient(ask_clarify=False)
        with client.stream(
            "POST", f"/runs/{run_id}/resume", headers=headers, json={"answers": ["the retrieval layer"]}
        ) as resp:
            assert resp.status_code == 200
            second = [json.loads(line) for line in resp.iter_lines() if line]

        final = next(e for e in second if e.get("type") == "run" and e.get("event") == "result")
        assert final["status"] == "done"
        assert final["verdict"] == "plausible"
        assert final["evaluation"]["verdict"] == "pass"

        got = client.get(f"/runs/{run_id}", headers=headers).json()
        assert got["run"]["status"] == "done"
        assert got["state"]["next"] == []
        assert any(e.get("event") == "result" for e in got["events"])

        listed = client.get("/runs", headers=headers).json()
        assert any(row["id"] == run_id for row in listed)

        # resuming a finished run is a conflict, not a crash
        assert client.post(f"/runs/{run_id}/resume", headers=headers, json={"answers": ["x"]}).status_code == 409
