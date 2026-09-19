"""Idea understanding handshake (M2)."""

import json

import pytest

from app.feasibility.understand import understand_idea


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

        async def _create(**kwargs):
            self.calls.append(kwargs)
            return type(
                "R",
                (),
                {"choices": [type("C", (), {"message": type("M", (), {"content": json.dumps(payload)})()})()]},
            )()

        self.chat = type("C", (), {"completions": type("CC", (), {"create": staticmethod(_create)})()})()


class _FailingClient:
    def __init__(self):
        async def _create(**kwargs):
            raise RuntimeError("LLM down")

        self.chat = type("C", (), {"completions": type("CC", (), {"create": staticmethod(_create)})()})()


@pytest.mark.asyncio
async def test_understand_returns_ready_understanding():
    client = _FakeClient(
        {
            "ready": True,
            "understanding": {
                "interpretation": "add a TTL cache to the retrieval layer",
                "problem": "repeated embeds are slow",
                "expected_outcome": "faster repeated queries",
                "assumptions": ["cache is in-process"],
                "hypotheses": ["hit rate > 50% on repeated queries"],
                "ambiguities": [],
            },
            "questions": [],
        }
    )
    result = await understand_idea(client, "add caching to retrieval")

    assert result["ready"] is True
    assert result["questions"] is None
    assert result["understanding"].interpretation.startswith("add a TTL cache")
    assert result["understanding"].hypotheses == ["hit rate > 50% on repeated queries"]
    assert client.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_understand_asks_clarifying_questions():
    client = _FakeClient(
        {
            "ready": False,
            "understanding": {"interpretation": "unclear"},
            "questions": ["Which module?", "What is the success criterion?"],
        }
    )
    result = await understand_idea(client, "make it better")
    assert result["ready"] is False
    assert result["understanding"] is None
    assert len(result["questions"]) == 2


@pytest.mark.asyncio
async def test_understand_answers_force_ready():
    client = _FakeClient(
        {
            "ready": False,  # LLM forgot to flip, answers must force it
            "understanding": {"interpretation": "add cache to rag.retrieval"},
            "questions": [],
        }
    )
    result = await understand_idea(client, "add caching", answers=["cache the retrieval layer"])
    assert result["ready"] is True
    assert result["understanding"] is not None


@pytest.mark.asyncio
async def test_understand_never_raises():
    result = await understand_idea(_FailingClient(), "broken idea")
    assert result["ready"] is True
    assert result["understanding"] is not None
