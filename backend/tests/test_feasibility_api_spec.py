"""Spec-driven API tests for /feasibility endpoints — auth, 400s, NDJSON, offline e2e (M2)."""

import json

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


def test_feasibility_requires_auth(api_client):
    client, _ = api_client
    assert client.get("/feasibility/runs").status_code == 401
    assert client.post("/feasibility/analyze", json={"idea": "x", "repo_id": "r"}).status_code == 401
    assert client.post("/feasibility/understand", json={"idea": "x"}).status_code == 401


def test_feasibility_validates_requests(api_client):
    client, _ = api_client
    assert client.post("/feasibility/analyze", headers=_headers(), json={"idea": "x"}).status_code == 422
    assert client.post("/feasibility/analyze", headers=_headers(), json={"idea": "", "repo_id": "r"}).status_code == 422
    assert client.post("/feasibility/understand", headers=_headers(), json={"idea": ""}).status_code == 422


def test_feasibility_analyze_unknown_repo_is_400(api_client, monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setattr("app.feasibility.pipeline.db.repo_get", AsyncMock(return_value=None))
    client, _ = api_client
    resp = client.post(
        "/feasibility/analyze", headers=_headers(), json={"idea": "add caching", "repo_id": "missing"}
    )
    assert resp.status_code == 400
    assert "not indexed" in resp.json()["detail"]


def test_feasibility_understand_returns_json(api_client):
    client, app = api_client
    payload = {
        "ready": False,
        "understanding": {"interpretation": "?"},
        "questions": ["Which module?"],
    }
    app.state.client = _RoutingClient({"understand": payload})
    resp = client.post("/feasibility/understand", headers=_headers(), json={"idea": "make it better"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is False
    assert body["questions"] == ["Which module?"]


_UNDERSTAND = {
    "ready": True,
    "understanding": {
        "interpretation": "add caching to the engine",
        "problem": "slow",
        "expected_outcome": "faster",
        "assumptions": [],
        "hypotheses": [],
        "ambiguities": [],
    },
    "questions": [],
}
_ENRICH = {
    "extension_points": [{"path": "pkg/core.py", "symbol": "Engine", "reason": "state owner"}],
    "data_flow": "none",
    "data_available": [],
    "data_missing": [],
    "data_cleanliness": "unknown",
}
_CLAIMS = {
    "claims": [
        {
            "statement": "Engine owns the state to cache.",
            "confidence": "high",
            "evidence": [{"path": "pkg/core.py", "start_line": 1, "end_line": 2}],
        }
    ]
}
_JUDGE = {
    "dimensions": {
        d: {"plausible": True, "rationale": "evidence", "blockers": [], "missing_evidence": []}
        for d in ("technical", "integration", "data")
    },
    "verdict": "plausible",
    "blockers": [],
    "alternatives": [],
    "reframed_idea": None,
}


class _Usage:
    prompt_tokens = 10
    completion_tokens = 5


class _RoutingClient:
    def __init__(self, overrides: dict | None = None):
        self.overrides = overrides or {}

        async def _create(**kwargs):
            system = kwargs["messages"][0]["content"]
            if "idea clarifier" in system:
                payload = self.overrides.get("understand", _UNDERSTAND)
            elif "codebase analyst" in system:
                payload = self.overrides.get("enrich", _ENRICH)
            elif "feasibility analyst" in system:
                payload = self.overrides.get("claims", _CLAIMS)
            elif "feasibility judge" in system:
                payload = self.overrides.get("judge", _JUDGE)
            else:
                raise AssertionError(f"unexpected system prompt: {system[:80]}")
            return type(
                "R",
                (),
                {
                    "choices": [type("C", (), {"message": type("M", (), {"content": json.dumps(payload)})()})()],
                    "usage": _Usage(),
                },
            )()

        self.chat = type("C", (), {"completions": type("CC", (), {"create": staticmethod(_create)})()})()


def test_offline_feasibility_e2e(tmp_path, monkeypatch):
    """M2 acceptance: index a repo offline -> analyze -> cited report -> persisted run."""
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

        app.state.client = _RoutingClient()

        with client.stream(
            "POST",
            "/feasibility/analyze",
            headers=headers,
            json={"idea": "add caching to the engine", "repo_id": repo_id},
        ) as resp:
            assert resp.status_code == 200
            assert "x-ndjson" in resp.headers.get("content-type", "")
            events = [json.loads(line) for line in resp.iter_lines() if line]

        kinds = [e["event"] for e in events if e.get("type") == "feasibility"]
        assert kinds[0] == "start"
        assert kinds.count("claim") == 3
        assert kinds[-1] == "report"

        report = events[-1]["data"]
        assert report["verdict"] == "plausible"
        assert report["blockers"] == []
        assert all(c["status"] == "supported" for c in report["claims"])
        assert all(any(r["valid"] for r in c["evidence"]) for c in report["claims"])
        assert report["usage"]["calls"] >= 4

        runs = client.get("/feasibility/runs", headers=headers).json()
        assert any(row["id"] == report["run_id"] for row in runs)

        got = client.get(f"/feasibility/runs/{report['run_id']}", headers=headers).json()
        assert got["run"]["verdict"] == "plausible"
        assert got["report"]["report"]["verdict"] == "plausible"
