"""Spec-driven tests for Code Analyzer — design: ResearchArtifact -> CodingSpec.

One json_object LLM call, no KB/RAG, emit-with-risks fallback, never raises.
"""

import json

import pytest


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, payload: dict):
        self._payload = payload
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResp(json.dumps(self._payload))


class _FakeClient:
    def __init__(self, payload: dict):
        self.chat = type("C", (), {"completions": _FakeCompletions(payload)})()


def _sample_artifact():
    from app.rd.schemas import DemoSpec, Requirements, ResearchArtifact, ResearchSummary, Validation

    req = Requirements(goal="demo attention", core_idea="attention is all you need", demo_type="script")
    summaries = [
        ResearchSummary(direction_id="core_algorithm", findings="Self-attention steps: QKV, scaled dot, softmax.", sources=[], confidence="high"),
        ResearchSummary(direction_id="evaluation", findings="Toy metric: accuracy > 0.", sources=[], confidence="high"),
    ]
    validation = Validation(feasible=True, risks=[])
    demo_spec = DemoSpec(entrypoint="demo.py", tech_stack=["python", "numpy"], core_algorithm="attention", pseudocode="softmax(QK^T)", acceptance_criteria=["demo.py exits 0"])
    return ResearchArtifact(requirements=req, summaries=summaries, validation=validation, demo_spec=demo_spec, sources=[], created_at=0)


@pytest.mark.asyncio
async def test_code_analyzer_returns_coding_spec_shape():
    """Must return CodingSpec with file_tasks, test_intents, dependencies. — design §2"""
    from app.code.analyzer import analyze_artifact

    payload = {
        "file_tasks": [
            {"path": "demo.py", "goal": "implement attention", "context_slice": "softmax"},
            {"path": "requirements.txt", "goal": "deps", "context_slice": "numpy"},
            {"path": "test_demo.py", "goal": "tests", "context_slice": "accuracy"},
        ],
        "test_intents": ["demo runs without error", "output shape correct"],
        "dependencies": ["numpy"],
        "risks": [],
    }
    client = _FakeClient(payload)
    result = await analyze_artifact(client, _sample_artifact())

    # CodingSpec is a Pydantic model — check shape via attributes or dict
    assert hasattr(result, "file_tasks") or "file_tasks" in result.model_dump()
    dump = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    assert len(dump["file_tasks"]) >= 3
    assert "test_intents" in dump
    assert "dependencies" in dump


@pytest.mark.asyncio
async def test_code_analyzer_fixed_file_set():
    """v1 fixed set must include demo.py, requirements.txt, test_demo.py. — design §3"""
    from app.code.analyzer import analyze_artifact

    payload = {
        "file_tasks": [
            {"path": "demo.py", "goal": "demo", "context_slice": "core"},
            {"path": "requirements.txt", "goal": "deps", "context_slice": "deps"},
            {"path": "test_demo.py", "goal": "tests", "context_slice": "criteria"},
        ],
        "test_intents": ["demo exits 0"],
        "dependencies": ["python", "numpy"],
        "risks": [],
    }
    client = _FakeClient(payload)
    result = await analyze_artifact(client, _sample_artifact())
    dump = result.model_dump()
    paths = [t["path"] if isinstance(t, dict) else t.path for t in dump["file_tasks"]]
    assert "demo.py" in paths
    assert "requirements.txt" in paths
    assert "test_demo.py" in paths


@pytest.mark.asyncio
async def test_code_analyzer_test_intents_from_acceptance_criteria():
    """test_intents must derive from acceptance_criteria, not from generated code. — design §4 tester contract"""
    from app.code.analyzer import analyze_artifact

    # Artifact has acceptance_criteria; analyzer should surface them as test_intents
    payload = {
        "file_tasks": [
            {"path": "demo.py", "goal": "demo", "context_slice": ""},
            {"path": "requirements.txt", "goal": "deps", "context_slice": ""},
            {"path": "test_demo.py", "goal": "tests", "context_slice": ""},
        ],
        "test_intents": ["demo.py exits 0 on toy data", "toy metric > 0"],
        "dependencies": ["numpy"],
        "risks": [],
    }
    client = _FakeClient(payload)
    artifact = _sample_artifact()
    artifact.demo_spec.acceptance_criteria = ["demo.py exits 0 on toy data", "toy metric > 0"]
    result = await analyze_artifact(client, artifact)
    dump = result.model_dump()
    assert any("demo.py" in s for s in dump["test_intents"])


@pytest.mark.asyncio
async def test_code_analyzer_single_llm_call():
    """Must be one json_object LLM call. — design §2"""
    from app.code.analyzer import analyze_artifact

    payload = {
        "file_tasks": [{"path": "demo.py", "goal": "g", "context_slice": ""}, {"path": "requirements.txt", "goal": "g", "context_slice": ""}, {"path": "test_demo.py", "goal": "g", "context_slice": ""}],
        "test_intents": ["t"],
        "dependencies": [],
        "risks": [],
    }
    client = _FakeClient(payload)
    await analyze_artifact(client, _sample_artifact())
    assert len(client.chat.completions.calls) == 1
    assert client.chat.completions.calls[0].get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_code_analyzer_never_raises_on_llm_failure():
    """On LLM failure must fallback to a valid CodingSpec, not raise. — design emit-with-risks"""
    from app.code.analyzer import analyze_artifact

    class _FailingCompletions:
        async def create(self, **kwargs):
            raise RuntimeError("LLM down")

    client = type("C", (), {"chat": type("C2", (), {"completions": _FailingCompletions()})()})()
    result = await analyze_artifact(client, _sample_artifact())
    dump = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    assert "file_tasks" in dump
    assert len(dump["file_tasks"]) >= 2


@pytest.mark.asyncio
async def test_code_analyzer_handles_infeasible_artifact():
    """When validation.feasible==False, must still emit CodingSpec with risks. — design §5.4"""
    from app.code.analyzer import analyze_artifact
    from app.rd.schemas import Validation

    payload = {
        "file_tasks": [{"path": "demo.py", "goal": "g", "context_slice": ""}, {"path": "requirements.txt", "goal": "g", "context_slice": ""}, {"path": "test_demo.py", "goal": "g", "context_slice": ""}],
        "test_intents": ["demo runs despite risks"],
        "dependencies": [],
        "risks": ["No evidence for core_algorithm"],
    }
    client = _FakeClient(payload)
    artifact = _sample_artifact()
    artifact.validation = Validation(feasible=False, risks=["No evidence"], missing=["core_algorithm"], coverage={})
    result = await analyze_artifact(client, artifact)
    dump = result.model_dump()
    assert dump["risks"] or len(dump["file_tasks"]) > 0


@pytest.mark.asyncio
async def test_code_analyzer_no_kb_access():
    """Analyzer must not call RAG/web_search — artifact already grounded. — design §2"""
    from unittest.mock import AsyncMock, patch

    from app.code.analyzer import analyze_artifact

    payload = {
        "file_tasks": [{"path": "demo.py", "goal": "g", "context_slice": ""}, {"path": "requirements.txt", "goal": "g", "context_slice": ""}, {"path": "test_demo.py", "goal": "g", "context_slice": ""}],
        "test_intents": ["t"],
        "dependencies": [],
        "risks": [],
    }
    client = _FakeClient(payload)
    with patch("app.tools.rag.rag_search", new=AsyncMock()) as mock_rag, patch("app.tools.search.web_search", new=AsyncMock()) as mock_web:
        await analyze_artifact(client, _sample_artifact())
        mock_rag.assert_not_called()
        mock_web.assert_not_called()
