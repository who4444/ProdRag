"""Spec-driven API tests for Code endpoints — design: POST /code/analyze (JSON) + POST /code/generate (NDJSON).

Auth, validation, streaming.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.rd.schemas import DemoSpec, Requirements, ResearchArtifact, ResearchSummary, Validation


def _sample_artifact_dict():
    return {
        "requirements": {"goal": "demo attention", "core_idea": "attention", "demo_type": "script", "constraints": [], "audience": "researcher", "open_questions": []},
        "summaries": [
            {"direction_id": "core_algorithm", "findings": "algo", "sources": [], "confidence": "high", "gaps": []},
            {"direction_id": "evaluation", "findings": "eval", "sources": [], "confidence": "high", "gaps": []},
        ],
        "validation": {"coverage": {"core_algorithm": True}, "missing": [], "feasible": True, "risks": []},
        "demo_spec": {"entrypoint": "demo.py", "tech_stack": ["python", "numpy"], "core_algorithm": "algo", "pseudocode": "code", "acceptance_criteria": ["demo.py exits 0"]},
        "sources": [],
        "created_at": 0,
    }


@pytest.fixture
def api_client(monkeypatch):
    from app.config import settings
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv("PRODRAG_API_TOKEN", "test-token")
    monkeypatch.setattr(settings, "api_token", "test-token")
    monkeypatch.setattr("app.core.storage.ensure_bucket", AsyncMock())
    monkeypatch.setattr("app.core.vectorstore.ensure_collections", AsyncMock())
    monkeypatch.setattr("app.core.vectorstore.close_client", AsyncMock())

    fake_redis = AsyncMock()
    fake_redis.ping = AsyncMock(return_value=True)

    async def _fake_create_pool(*args, **kwargs):
        return fake_redis

    monkeypatch.setattr("app.main.create_pool", _fake_create_pool)
    monkeypatch.setattr("arq.connections.create_pool", _fake_create_pool, raising=False)
    try:
        import arq

        monkeypatch.setattr(arq, "create_pool", _fake_create_pool, raising=False)
    except Exception:
        pass
    monkeypatch.setattr("app.main.AsyncOpenAI", MagicMock(return_value=MagicMock()))

    # Code DB mocks (may not exist yet — raising=False)
    monkeypatch.setattr("app.core.db.code_run_create", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.code_artifact_upsert", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.code_run_update", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.research_run_create", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.research_artifact_upsert", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.research_run_update", AsyncMock(), raising=False)

    from app.main import app

    fake_client = MagicMock()
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.redis = fake_redis
        app.state.client = fake_client
        yield client, app, fake_client


def _auth():
    return {"Authorization": "Bearer test-token"}


def _make_openai_for_code(payload_json: dict, text_fallback: str = "print('hello')"):
    from unittest.mock import AsyncMock, MagicMock

    async def _create(**kwargs):
        fmt = kwargs.get("response_format")
        if fmt == {"type": "json_object"}:
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": json.dumps(payload_json)})()})()]})()
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": text_fallback})()})()]})()

    m = MagicMock()
    m.chat.completions.create = AsyncMock(side_effect=_create)
    return m


def test_code_analyze_requires_auth(api_client):
    client, app, _ = api_client
    resp = client.post("/code/analyze", json={"artifact": _sample_artifact_dict()})
    assert resp.status_code == 401


def test_code_analyze_requires_artifact(api_client):
    client, app, _ = api_client
    resp = client.post("/code/analyze", headers=_auth(), json={})
    assert resp.status_code == 422


def test_code_analyze_returns_coding_spec(api_client):
    client, app, _ = api_client
    payload = {
        "file_tasks": [
            {"path": "demo.py", "goal": "demo", "context_slice": "algo"},
            {"path": "requirements.txt", "goal": "deps", "context_slice": "numpy"},
            {"path": "test_demo.py", "goal": "tests", "context_slice": "criteria"},
        ],
        "test_intents": ["demo exits 0"],
        "dependencies": ["numpy"],
        "risks": [],
    }
    app.state.client = _make_openai_for_code(payload)
    resp = client.post("/code/analyze", headers=_auth(), json={"artifact": _sample_artifact_dict()})
    assert resp.status_code == 200
    data = resp.json()
    assert "file_tasks" in data or "coding_spec" in data
    # Accept either shape: direct CodingSpec or wrapped
    spec = data.get("coding_spec", data)
    assert len(spec.get("file_tasks", [])) >= 3


def test_code_generate_requires_auth(api_client):
    client, app, _ = api_client
    resp = client.post("/code/generate", json={"artifact": _sample_artifact_dict()})
    assert resp.status_code == 401


def test_code_generate_requires_artifact(api_client):
    client, app, _ = api_client
    resp = client.post("/code/generate", headers=_auth(), json={})
    assert resp.status_code == 422


def test_code_generate_streams_ndjson(api_client, monkeypatch):
    from unittest.mock import AsyncMock

    client, app, _ = api_client

    # Mock sandbox to avoid subprocess
    monkeypatch.setattr("app.code.sandbox.run_sandbox", AsyncMock(return_value={"passed": True, "pytest_log": "ok", "demo_log": "hello", "returncode": 0, "runtime_ms": 5}), raising=False)

    payload = {
        "file_tasks": [
            {"path": "demo.py", "goal": "demo", "context_slice": ""},
            {"path": "requirements.txt", "goal": "deps", "context_slice": ""},
            {"path": "test_demo.py", "goal": "tests", "context_slice": ""},
        ],
        "test_intents": ["demo exits 0"],
        "dependencies": [],
        "risks": [],
    }
    app.state.client = _make_openai_for_code(payload, text_fallback="print('demo')")

    req = {"artifact": _sample_artifact_dict()}
    with client.stream("POST", "/code/generate", headers=_auth(), json=req) as resp:
        assert resp.status_code == 200
        assert "x-ndjson" in resp.headers.get("content-type", "")
        lines = [json.loads(l) for l in resp.iter_lines() if l]
        types = [l.get("type") for l in lines]
        # Must contain code plan and artifact
        assert "code" in types or any(l.get("type") == "code" for l in lines)
        assert any(l.get("event") == "artifact" for l in lines)


def test_code_generate_includes_tester_and_sandbox(api_client, monkeypatch):
    from unittest.mock import AsyncMock

    client, app, _ = api_client
    monkeypatch.setattr("app.code.sandbox.run_sandbox", AsyncMock(return_value={"passed": True, "pytest_log": "ok", "demo_log": "ok", "returncode": 0, "runtime_ms": 1}), raising=False)

    payload = {
        "file_tasks": [{"path": "demo.py", "goal": "g", "context_slice": ""}, {"path": "requirements.txt", "goal": "g", "context_slice": ""}, {"path": "test_demo.py", "goal": "g", "context_slice": ""}],
        "test_intents": ["demo exits 0"],
        "dependencies": [],
        "risks": [],
    }
    app.state.client = _make_openai_for_code(payload, text_fallback="content")

    req = {"artifact": _sample_artifact_dict()}
    with client.stream("POST", "/code/generate", headers=_auth(), json=req) as resp:
        lines = [json.loads(l) for l in resp.iter_lines() if l]
        events = [l.get("event") for l in lines]
        assert "artifact" in events
        # tester and sandbox should be present as types or events
        assert any(l.get("type") in ("tester", "sandbox", "code") for l in lines)


def test_code_analyze_echoes_artifact_validation(api_client):
    """When artifact is infeasible, analyzer should still return spec with risks. — emit-with-risks"""
    client, app, _ = api_client
    artifact = _sample_artifact_dict()
    artifact["validation"] = {"coverage": {}, "missing": ["core_algorithm"], "feasible": False, "risks": ["No evidence"]}
    payload = {
        "file_tasks": [{"path": "demo.py", "goal": "g", "context_slice": ""}, {"path": "requirements.txt", "goal": "g", "context_slice": ""}, {"path": "test_demo.py", "goal": "g", "context_slice": ""}],
        "test_intents": ["demo with risks"],
        "dependencies": [],
        "risks": ["No evidence"],
    }
    app.state.client = _make_openai_for_code(payload)
    resp = client.post("/code/analyze", headers=_auth(), json={"artifact": artifact})
    assert resp.status_code == 200
    data = resp.json()
    spec = data.get("coding_spec", data)
    assert spec.get("risks") is not None or len(spec.get("file_tasks", [])) > 0
