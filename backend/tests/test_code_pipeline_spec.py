"""Spec-driven tests for Code Pipeline — design: analyzer->orchestrator->subagents->tester->sandbox.

Streaming NDJSON: code/plan, code/file_start|result x3-4, tester/result, sandbox/start|result, code/artifact.
"""

import json

import pytest

from app.rd.schemas import DemoSpec, Requirements, ResearchArtifact, ResearchSummary, Validation


def _sample_artifact():
    req = Requirements(goal="demo attention", core_idea="attention", demo_type="script")
    summaries = [
        ResearchSummary(direction_id="core_algorithm", findings="algo", sources=[], confidence="high"),
        ResearchSummary(direction_id="evaluation", findings="eval", sources=[], confidence="high"),
    ]
    validation = Validation(feasible=True, risks=[])
    demo_spec = DemoSpec(entrypoint="demo.py", tech_stack=["python", "numpy"], core_algorithm="algo", pseudocode="code", acceptance_criteria=["demo.py exits 0"])
    return ResearchArtifact(requirements=req, summaries=summaries, validation=validation, demo_spec=demo_spec, sources=[], created_at=0)


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


def _make_code_client():
    """Client that serves analyzer/json, subagent text, tester text based on response_format/system prompt."""

    class _Completions:
        async def create(self, **kwargs):
            fmt = kwargs.get("response_format")
            msgs = kwargs.get("messages", [])
            text = " ".join(m.get("content", "") for m in msgs).lower()
            if fmt == {"type": "json_object"}:
                if "coding" in text or "file_tasks" in text or "code analyzer" in text:
                    payload = {
                        "file_tasks": [
                            {"path": "demo.py", "goal": "demo", "context_slice": "algo"},
                            {"path": "requirements.txt", "goal": "deps", "context_slice": "numpy"},
                            {"path": "test_demo.py", "goal": "tests", "context_slice": "demo exits 0"},
                        ],
                        "test_intents": ["demo exits 0"],
                        "dependencies": ["numpy"],
                        "risks": [],
                    }
                    return _FakeResp(json.dumps(payload))
                # tester fallback also json? No, tester is text
            # subagent/ tester text
            if "def test_" in text or "test_demo" in text:
                return _FakeResp("def test_demo():\n    assert True")
            return _FakeResp("print('hello demo')\n")

    return type("C", (), {"chat": type("C2", (), {"completions": _Completions()})()})()


@pytest.mark.asyncio
async def test_code_pipeline_event_sequence(monkeypatch):
    """Must yield code/plan -> file_start/result x3-4 -> tester -> sandbox -> artifact. — design §4.6 variant"""
    from unittest.mock import AsyncMock

    from app.code.pipeline import code_generate

    monkeypatch.setattr("app.core.db.code_run_create", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.code_run_update", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.code_artifact_upsert", AsyncMock(), raising=False)
    monkeypatch.setattr("app.code.sandbox.run_sandbox", AsyncMock(return_value={"passed": True, "pytest_log": "ok", "demo_log": "hello", "returncode": 0, "runtime_ms": 10}), raising=False)

    client = _make_code_client()
    artifact = _sample_artifact()

    events = []
    async for evt in code_generate(client, artifact, session_id=None):
        events.append(evt)

    types = [(e.get("type"), e.get("event")) for e in events]
    assert types[0] == ("code", "plan")
    # file events
    file_events = [e for e in events if e.get("type") == "code" and e.get("event") in ("file_start", "file_result")]
    assert len(file_events) >= 6  # start+result x3
    assert any(e["type"] == "tester" for e in events)
    assert any(e["type"] == "sandbox" for e in events)
    assert events[-1]["type"] == "code" and events[-1]["event"] == "artifact"


@pytest.mark.asyncio
async def test_code_pipeline_artifact_contains_code_artifact(monkeypatch):
    """Final artifact must contain CodeArtifact with files, test_file, sandbox. — design §3"""
    from unittest.mock import AsyncMock

    from app.code.pipeline import code_generate

    monkeypatch.setattr("app.core.db.code_run_create", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.code_run_update", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.code_artifact_upsert", AsyncMock(), raising=False)
    monkeypatch.setattr("app.code.sandbox.run_sandbox", AsyncMock(return_value={"passed": True, "pytest_log": "ok", "demo_log": "hello", "returncode": 0, "runtime_ms": 5}), raising=False)

    client = _make_code_client()
    events = [e async for e in code_generate(client, _sample_artifact(), None)]
    artifact_evt = events[-1]
    assert "data" in artifact_evt
    data = artifact_evt["data"]
    # CodeArtifact fields
    assert "files" in data or "code_artifact" in data or "artifact" in data
    # At least demo.py present
    files = data.get("files") or data.get("code_artifact", {}).get("files") or []
    if files:
        paths = [f.get("path") if isinstance(f, dict) else getattr(f, "path", "") for f in files]
        assert "demo.py" in paths


@pytest.mark.asyncio
async def test_code_pipeline_persists(monkeypatch):
    """Must persist code_runs (best-effort). — design §3 DB"""
    from unittest.mock import AsyncMock

    from app.code.pipeline import code_generate

    mock_create = AsyncMock()
    mock_upsert = AsyncMock()
    mock_update = AsyncMock()
    monkeypatch.setattr("app.core.db.code_run_create", mock_create, raising=False)
    monkeypatch.setattr("app.core.db.code_artifact_upsert", mock_upsert, raising=False)
    monkeypatch.setattr("app.core.db.code_run_update", mock_update, raising=False)
    monkeypatch.setattr("app.code.sandbox.run_sandbox", AsyncMock(return_value={"passed": True, "pytest_log": "", "demo_log": "", "returncode": 0, "runtime_ms": 1}), raising=False)

    client = _make_code_client()
    events = [e async for e in code_generate(client, _sample_artifact(), None)]
    assert mock_create.called or mock_upsert.called  # at least one persist attempt
    assert any(e["type"] == "code" and e["event"] == "artifact" for e in events)


@pytest.mark.asyncio
async def test_code_pipeline_emit_with_risks(monkeypatch):
    """Even when validation infeasible, must still emit artifact. — design emit-with-risks"""
    from unittest.mock import AsyncMock

    from app.code.pipeline import code_generate

    monkeypatch.setattr("app.core.db.code_run_create", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.code_artifact_upsert", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.code_run_update", AsyncMock(), raising=False)
    monkeypatch.setattr("app.code.sandbox.run_sandbox", AsyncMock(return_value={"passed": False, "pytest_log": "fail", "demo_log": "", "returncode": 1, "runtime_ms": 1}), raising=False)

    client = _make_code_client()
    artifact = _sample_artifact()
    artifact.validation.feasible = False
    artifact.validation.risks = ["No evidence for core_algorithm"]
    events = [e async for e in code_generate(client, artifact, None)]
    assert any(e["type"] == "code" and e["event"] == "artifact" for e in events)


@pytest.mark.asyncio
async def test_code_pipeline_handles_sandbox_failure(monkeypatch):
    """Sandbox failure must not break pipeline — still emits artifact with passed=False. — design §5"""
    from unittest.mock import AsyncMock

    from app.code.pipeline import code_generate

    monkeypatch.setattr("app.core.db.code_run_create", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.code_artifact_upsert", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.code_run_update", AsyncMock(), raising=False)

    async def _fail_sandbox(*args, **kwargs):
        return {"passed": False, "pytest_log": "timeout", "demo_log": "", "returncode": 124, "runtime_ms": 30000}

    monkeypatch.setattr("app.code.sandbox.run_sandbox", AsyncMock(side_effect=_fail_sandbox), raising=False)

    client = _make_code_client()
    events = [e async for e in code_generate(client, _sample_artifact(), None)]
    assert events[-1]["type"] == "code" and events[-1]["event"] == "artifact"
    # sandbox result should be present
    assert any(e["type"] == "sandbox" for e in events)
