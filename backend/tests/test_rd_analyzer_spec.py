"""Spec-driven tests for Analyzer — PROJECT.md §4.1, API §2.4.

Covers: second-request handshake, Requirements shape, demo_type enum,
1-3 questions, max 2 rounds, no KB access, plain JSON (not streamed).

These tests are derived purely from spec — they do not inspect implementation internals.
"""

import json

import pytest

from app.rd.schemas import AnalyzeRequest, Requirements


# ---------------------------------------------------------------------------
# Helpers: fake OpenAI client that returns controllable JSON
# ---------------------------------------------------------------------------

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


class _FakeChat:
    def __init__(self, payload: dict):
        self.completions = _FakeCompletions(payload)


class _FakeClient:
    def __init__(self, payload: dict):
        self.chat = _FakeChat(payload)


# ---------------------------------------------------------------------------
# Analyzer unit tests — spec §4.1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyzer_vague_idea_returns_questions():
    """Vague idea without answers → ready=false with 1-3 questions. §4.1"""
    from app.rd.analyzer import analyze_idea

    vague_payload = {
        "ready": False,
        "requirements": {
            "goal": "demo",
            "core_idea": "vague",
            "demo_type": "script",
            "constraints": [],
            "audience": "researcher",
            "open_questions": ["what demo?"],
        },
        "questions": ["What dataset should the demo use?", "What interaction do you expect?"],
    }
    client = _FakeClient(vague_payload)
    result = await analyze_idea(client, idea="make a demo", answers=None)

    assert result["ready"] is False
    assert result["questions"] is not None
    assert 1 <= len(result["questions"]) <= 3
    assert result["requirements"] is None  # not ready → no requirements
    assert result["draft"] is not None  # draft may be returned when not ready


@pytest.mark.asyncio
async def test_analyzer_precise_idea_returns_requirements():
    """Precise idea → ready=true with full Requirements. §4.1"""
    from app.rd.analyzer import analyze_idea

    precise_payload = {
        "ready": True,
        "requirements": {
            "goal": "demo attention mechanism on toy data",
            "core_idea": "Self-attention computes weighted sum of values",
            "demo_type": "script",
            "constraints": ["python-only"],
            "audience": "researcher",
            "open_questions": [],
        },
    }
    client = _FakeClient(precise_payload)
    result = await analyze_idea(
        client,
        idea="Turn Attention Is All You Need core algorithm into a minimal python script demo on toy data for researchers",
        answers=None,
    )

    assert result["ready"] is True
    assert result["questions"] is None
    req = result["requirements"]
    assert isinstance(req, Requirements)
    assert req.goal
    assert req.core_idea
    assert req.demo_type in ("script", "notebook")  # v1 enum only
    assert isinstance(req.constraints, list)
    assert req.audience in ("researcher", "student", "engineer")


@pytest.mark.asyncio
async def test_analyzer_second_request_forces_ready():
    """Second request with answers must return ready=true (cap at 2 rounds). §4.1"""
    from app.rd.analyzer import analyze_idea

    # LLM erroneously still says ready=false, but analyzer must force ready when answers supplied
    payload_still_vague = {
        "ready": False,
        "requirements": {
            "goal": "demo with answers",
            "core_idea": "filled via answers",
            "demo_type": "script",
            "constraints": [],
            "audience": "researcher",
            "open_questions": [],
        },
        "questions": ["leftover question"],
    }
    client = _FakeClient(payload_still_vague)
    result = await analyze_idea(client, idea="vague idea", answers=["Use toy data", "Script demo"])

    assert result["ready"] is True, "second-request handshake must infer defaults and return ready=true"
    assert result["requirements"] is not None


@pytest.mark.asyncio
async def test_analyzer_requirements_shape_validated():
    """Requirements must contain all spec fields and be a valid model. §4.1"""
    from app.rd.analyzer import analyze_idea

    payload = {
        "ready": True,
        "requirements": {
            "goal": "g",
            "core_idea": "c",
            "demo_type": "notebook",
            "constraints": ["no GPU"],
            "audience": "student",
            "open_questions": [],
        },
    }
    client = _FakeClient(payload)
    result = await analyze_idea(client, idea="precise idea", answers=None)
    req = result["requirements"]
    # All fields present and typed
    assert hasattr(req, "goal") and isinstance(req.goal, str)
    assert hasattr(req, "core_idea") and isinstance(req.core_idea, str)
    assert hasattr(req, "demo_type")
    assert hasattr(req, "constraints") and isinstance(req.constraints, list)
    assert hasattr(req, "audience")
    assert hasattr(req, "open_questions") and isinstance(req.open_questions, list)


@pytest.mark.asyncio
async def test_analyzer_demo_type_enum_v1():
    """demo_type must be script|notebook in v1; gradio/api are v2 and must be rejected/normalized. §4.1 non-goals"""
    from app.rd.analyzer import analyze_idea

    payload_gradio = {
        "ready": True,
        "requirements": {
            "goal": "g",
            "core_idea": "c",
            "demo_type": "gradio",  # v2, should be normalized to script/notebook
            "constraints": [],
            "audience": "researcher",
            "open_questions": [],
        },
    }
    client = _FakeClient(payload_gradio)
    result = await analyze_idea(client, idea="idea asking for gradio", answers=None)
    req = result["requirements"]
    assert req.demo_type in ("script", "notebook"), "v1 must not emit gradio/demo v2 types"


@pytest.mark.asyncio
async def test_analyzer_no_kb_access():
    """Analyzer must not call RAG/knowledge base — spec explicitly says no KB. §4.1"""
    from unittest.mock import AsyncMock, patch

    from app.rd.analyzer import analyze_idea

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
    client = _FakeClient(payload)

    # If analyzer called rag_search, this mock would be hit
    with patch("app.tools.rag.rag_search", new=AsyncMock()) as mock_rag, patch(
        "app.rag.retrieval.search_kb", new=AsyncMock()
    ) as mock_search:
        await analyze_idea(client, idea="test idea", answers=None)
        mock_rag.assert_not_called()
        mock_search.assert_not_called()


@pytest.mark.asyncio
async def test_analyzer_single_llm_call():
    """Analyzer must be one LLM call (ponytail) — not a loop. §4.1"""
    from app.rd.analyzer import analyze_idea

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
    client = _FakeClient(payload)
    await analyze_idea(client, idea="idea", answers=None)
    assert len(client.chat.completions.calls) == 1
    # must request json_object
    assert client.chat.completions.calls[0].get("response_format") == {"type": "json_object"}


def test_analyzer_request_schema_validation():
    """AnalyzeRequest requires idea min_length 1. Spec §2.4"""
    with pytest.raises(Exception):
        AnalyzeRequest(idea="")
    # valid
    req = AnalyzeRequest(idea="hello", answers=["a1"])
    assert req.idea == "hello"
    assert req.answers == ["a1"]


@pytest.mark.asyncio
async def test_analyzer_handles_llm_failure_gracefully():
    """Analyzer must fallback to a valid Requirements even if LLM fails. Spec robustness."""
    from app.rd.analyzer import analyze_idea

    class _FailingCompletions:
        async def create(self, **kwargs):
            raise RuntimeError("LLM down")

    class _FailingClient:
        chat = type("C", (), {"completions": _FailingCompletions()})()

    result = await analyze_idea(_FailingClient(), idea="fallback test", answers=None)
    # Must still return a usable shape, not raise
    assert result["ready"] is True
    assert isinstance(result["requirements"], Requirements)
