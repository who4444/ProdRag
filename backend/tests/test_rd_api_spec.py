"""Spec-driven API tests for R&D endpoints — PROJECT.md §2.4, §4.8.

Covers: auth, request validation, second-request handshake, NDJSON streaming,
event order, fixed taxonomy, emit-with-risks.
"""

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(monkeypatch):
    """TestClient with mocked auth and external deps."""
    monkeypatch.setenv("PRODRAG_API_TOKEN", "test-token")
    from app.config import settings

    monkeypatch.setattr(settings, "api_token", "test-token")

    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setattr("app.core.storage.ensure_bucket", AsyncMock())
    monkeypatch.setattr("app.core.vectorstore.ensure_collections", AsyncMock())
    monkeypatch.setattr("app.core.vectorstore.close_client", AsyncMock())

    fake_redis = AsyncMock()
    fake_redis.ping = AsyncMock(return_value=True)
    fake_redis.hset = AsyncMock()
    fake_redis.rpush = AsyncMock()
    fake_redis.ltrim = AsyncMock()
    fake_redis.expire = AsyncMock()

    async def _fake_create_pool(*args, **kwargs):
        return fake_redis

    monkeypatch.setattr("app.main.create_pool", _fake_create_pool)
    monkeypatch.setattr("arq.connections.create_pool", _fake_create_pool, raising=False)
    try:
        import arq  # noqa

        monkeypatch.setattr(arq, "create_pool", _fake_create_pool, raising=False)
    except Exception:
        pass

    monkeypatch.setattr("app.main.AsyncOpenAI", MagicMock(return_value=MagicMock()))

    from app.main import app

    fake_client = MagicMock()

    monkeypatch.setattr("app.core.db.research_run_create", AsyncMock())
    monkeypatch.setattr("app.core.db.research_artifact_upsert", AsyncMock())
    monkeypatch.setattr("app.core.db.research_run_update", AsyncMock())
    monkeypatch.setattr("app.core.db.conversation_create", AsyncMock(return_value="test-session"))
    monkeypatch.setattr("app.core.db.message_add", AsyncMock())
    monkeypatch.setattr("app.core.db.messages_for", AsyncMock(return_value=[]))

    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.redis = fake_redis
        app.state.client = fake_client
        yield client, app, fake_client


def _auth_headers():
    return {"Authorization": "Bearer test-token"}


def _make_openai_mock(payload: dict):
    """Return a MagicMock client whose chat.completions.create returns payload as JSON."""
    from unittest.mock import AsyncMock, MagicMock

    async def _create(**kwargs):
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": json.dumps(payload)})()})()]})()

    mock = MagicMock()
    mock.chat.completions.create = AsyncMock(side_effect=_create)
    return mock


def _make_openai_mock_text(text: str):
    from unittest.mock import AsyncMock, MagicMock

    async def _create(**kwargs):
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": text})()})()]})()

    mock = MagicMock()
    mock.chat.completions.create = AsyncMock(side_effect=_create)
    return mock


# ---------------------------------------------------------------------------
# /rd/analyze — spec §2.4, §4.1
# ---------------------------------------------------------------------------


def test_rd_analyze_requires_auth(api_client):
    client, app, _ = api_client
    resp = client.post("/rd/analyze", json={"idea": "hello"})
    assert resp.status_code == 401


def test_rd_analyze_requires_idea(api_client):
    client, app, _ = api_client
    resp = client.post("/rd/analyze", headers=_auth_headers(), json={"idea": ""})
    assert resp.status_code == 422
    resp2 = client.post("/rd/analyze", headers=_auth_headers(), json={})
    assert resp2.status_code == 422


def test_rd_analyze_returns_json_with_ready(api_client, monkeypatch):
    """Must return {ready, requirements/questions} as plain JSON, not streamed. §4.1"""
    client, app, _ = api_client

    payload = {
        "ready": True,
        "requirements": {
            "goal": "demo attention",
            "core_idea": "attention is all you need",
            "demo_type": "script",
            "constraints": [],
            "audience": "researcher",
            "open_questions": [],
        },
    }
    app.state.client = _make_openai_mock(payload)

    resp = client.post("/rd/analyze", headers=_auth_headers(), json={"idea": "Turn attention paper into demo"})
    assert resp.status_code == 200
    data = resp.json()
    assert "ready" in data
    assert isinstance(data["ready"], bool)
    assert data["ready"] is True
    assert "requirements" in data
    assert data["requirements"]["demo_type"] in ("script", "notebook")


def test_rd_analyze_second_request_handshake(api_client, monkeypatch):
    """First call ready=false with questions, second call with answers forces ready=true. §4.1"""
    client, app, _ = api_client

    vague_payload = {
        "ready": False,
        "requirements": {
            "goal": "demo",
            "core_idea": "vague",
            "demo_type": "script",
            "constraints": [],
            "audience": "researcher",
            "open_questions": [],
        },
        "questions": ["What dataset?", "What demo type?"],
    }
    app.state.client = _make_openai_mock(vague_payload)

    resp1 = client.post("/rd/analyze", headers=_auth_headers(), json={"idea": "make a demo"})
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["ready"] is False
    assert 1 <= len(data1["questions"]) <= 3

    precise_payload = {
        "ready": True,
        "requirements": {
            "goal": "demo attention on toy data",
            "core_idea": "self-attention",
            "demo_type": "script",
            "constraints": ["python-only"],
            "audience": "researcher",
            "open_questions": [],
        },
    }
    app.state.client = _make_openai_mock(precise_payload)

    resp2 = client.post(
        "/rd/analyze", headers=_auth_headers(), json={"idea": "make a demo", "answers": ["toy data", "script"]}
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["ready"] is True
    assert "requirements" in data2


def test_rd_analyze_session_id_echo(api_client, monkeypatch):
    """Should echo session_id when provided. Spec §4.1 input includes session_id."""
    client, app, _ = api_client

    payload = {
        "ready": True,
        "requirements": {
            "goal": "g",
            "core_idea": "c",
            "demo_type": "script",
            "constraints": [],
            "audience": "researcher",
            "open_questions": [],
        },
    }
    app.state.client = _make_openai_mock(payload)

    resp = client.post("/rd/analyze", headers=_auth_headers(), json={"idea": "hello", "session_id": "sess-123"})
    assert resp.status_code == 200
    assert resp.json().get("session_id") == "sess-123"


# ---------------------------------------------------------------------------
# /rd/research — spec §2.4, §4.6, §4.8
# ---------------------------------------------------------------------------


def test_rd_research_requires_auth(api_client):
    client, app, _ = api_client
    resp = client.post("/rd/research", json={"requirements": {"goal": "g", "core_idea": "c", "demo_type": "script"}})
    assert resp.status_code == 401


def test_rd_research_requires_requirements(api_client):
    client, app, _ = api_client
    resp = client.post("/rd/research", headers=_auth_headers(), json={})
    assert resp.status_code == 422
    resp2 = client.post("/rd/research", headers=_auth_headers(), json={"requirements": {"goal": "", "core_idea": ""}})
    assert resp2.status_code in (200, 422)


def test_rd_research_streams_ndjson_and_event_order(api_client, monkeypatch):
    """Must stream NDJSON with orchestrator/plan → subagent/* → validator → artifact. §4.8"""
    from unittest.mock import AsyncMock

    client, app, _ = api_client

    mock_rag = AsyncMock(return_value={"text": [{"page": 1, "score": 0.9, "text": "evidence", "source": "paper.pdf"}], "images": []})
    monkeypatch.setattr("app.rd.subagent.rag_search", mock_rag)
    monkeypatch.setattr("app.tools.rag.rag_search", mock_rag, raising=False)

    from unittest.mock import MagicMock

    async def _create(**kwargs):
        fmt = kwargs.get("response_format")
        msgs = kwargs.get("messages", [])
        system = " ".join(m.get("content", "") for m in msgs if m.get("role") == "system")
        if fmt == {"type": "json_object"} and "validator" in system.lower():
            payload = {"coverage": {"core_algorithm": True, "datasets": True, "baselines": True, "implementation_details": True, "evaluation": True}, "missing": [], "feasible": True, "risks": []}
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": json.dumps(payload)})()})()]})()
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "Findings [1]."})()})()]})()

    fake_openai = MagicMock()
    fake_openai.chat.completions.create = AsyncMock(side_effect=_create)
    app.state.client = fake_openai

    req_body = {
        "requirements": {"goal": "demo attention", "core_idea": "attention is all you need", "demo_type": "script", "constraints": [], "audience": "researcher", "open_questions": []},
        "idea": "demo attention",
    }
    with client.stream("POST", "/rd/research", headers=_auth_headers(), json=req_body) as resp:
        assert resp.status_code == 200
        assert "x-ndjson" in resp.headers.get("content-type", "")
        lines = [json.loads(line) for line in resp.iter_lines() if line]
        types = [l["type"] for l in lines]
        assert types[0] == "orchestrator"
        assert "subagent" in types
        assert "validator" in types
        assert types[-1] == "artifact"
        plan = next(l for l in lines if l["type"] == "orchestrator")
        assert len(plan["directions"]) == 5
        validator = next(l for l in lines if l["type"] == "validator")
        assert "feasible" in validator["validation"]


def test_rd_research_fixed_taxonomy(api_client, monkeypatch):
    """Directions must be the fixed 5 IDs. §4.2"""
    from unittest.mock import AsyncMock, MagicMock

    client, app, _ = api_client

    mock_rag = AsyncMock(return_value={"text": [], "images": []})
    monkeypatch.setattr("app.rd.subagent.rag_search", mock_rag)
    monkeypatch.setattr("app.tools.rag.rag_search", mock_rag, raising=False)

    async def _create(**kwargs):
        fmt = kwargs.get("response_format")
        if fmt == {"type": "json_object"}:
            payload = {"coverage": {}, "missing": [], "feasible": True, "risks": []}
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": json.dumps(payload)})()})()]})()
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "findings"})()})()]})()

    fake_openai = MagicMock()
    fake_openai.chat.completions.create = AsyncMock(side_effect=_create)
    app.state.client = fake_openai

    req_body = {"requirements": {"goal": "g", "core_idea": "c", "demo_type": "script", "constraints": [], "audience": "researcher", "open_questions": []}}
    with client.stream("POST", "/rd/research", headers=_auth_headers(), json=req_body) as resp:
        lines = [json.loads(line) for line in resp.iter_lines() if line]
        plan = next(l for l in lines if l["type"] == "orchestrator")
        ids = [d["id"] for d in plan["directions"]]
        assert ids == ["core_algorithm", "datasets", "baselines", "implementation_details", "evaluation"]


def test_rd_research_artifact_has_demo_spec(api_client, monkeypatch):
    """Artifact must contain demo_spec with entrypoint demo.py and acceptance_criteria. §4.5"""
    from unittest.mock import AsyncMock, MagicMock

    client, app, _ = api_client

    mock_rag = AsyncMock(return_value={"text": [], "images": []})
    monkeypatch.setattr("app.rd.subagent.rag_search", mock_rag)
    monkeypatch.setattr("app.tools.rag.rag_search", mock_rag, raising=False)

    async def _create(**kwargs):
        if kwargs.get("response_format") == {"type": "json_object"}:
            payload = {"coverage": {}, "missing": [], "feasible": False, "risks": ["risk1"]}
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": json.dumps(payload)})()})()]})()
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "findings"})()})()]})()

    fake_openai = MagicMock()
    fake_openai.chat.completions.create = AsyncMock(side_effect=_create)
    app.state.client = fake_openai

    req_body = {"requirements": {"goal": "g", "core_idea": "c", "demo_type": "script", "constraints": [], "audience": "researcher", "open_questions": []}}
    with client.stream("POST", "/rd/research", headers=_auth_headers(), json=req_body) as resp:
        lines = [json.loads(line) for line in resp.iter_lines() if line]
        artifact = next(l for l in lines if l["type"] == "artifact")
        assert artifact["data"]["demo_spec"]["entrypoint"] == "demo.py"
        assert len(artifact["data"]["demo_spec"]["acceptance_criteria"]) >= 1
        assert artifact["data"]["validation"]["feasible"] is False
        assert artifact["data"]["validation"]["risks"]
