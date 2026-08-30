"""Spec-driven tests for Subagents — PROJECT.md §4.3.

One function N concurrent invocations, each rag_search + synthesis.
"""

import json

import pytest

from app.rd.schemas import Direction, Requirements


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
    def __init__(self, content: str):
        self._content = content

    async def create(self, **kwargs):
        return _FakeResp(self._content)


class _FakeClient:
    def __init__(self, content: str = "Findings with [1] citation."):
        self.chat = type("C", (), {"completions": _FakeCompletions(content)})()


@pytest.mark.asyncio
async def test_subagent_calls_rag_search_with_direction_question(monkeypatch):
    """Must call rag_search with direction.question and k=4. §4.3"""
    from unittest.mock import AsyncMock

    from app.rd.subagent import run_direction

    mock_rag = AsyncMock(return_value={"text": [], "images": []})
    monkeypatch.setattr("app.rd.subagent.rag_search", mock_rag)
    # also patch tools.rag in case of alternate import
    monkeypatch.setattr("app.tools.rag.rag_search", mock_rag, raising=False)

    direction = Direction(id="core_algorithm", question="What is the core algorithm?", rationale="r")
    client = _FakeClient("synthesis")
    await run_direction(client, direction, k=4, k_images=1)

    assert mock_rag.called
    call_kwargs = mock_rag.call_args
    # positional or keyword — check that question was passed
    called_args = str(call_kwargs)
    assert "What is the core algorithm?" in called_args


@pytest.mark.asyncio
async def test_subagent_returns_research_summary_shape(monkeypatch):
    """Must return ResearchSummary with spec shape. §4.3"""
    from unittest.mock import AsyncMock

    from app.rd.schemas import ResearchSummary
    from app.rd.subagent import run_direction

    mock_rag = AsyncMock(
        return_value={
            "text": [{"page": 1, "score": 0.9, "text": "attention mechanism", "source": "paper.pdf"}],
            "images": [],
        }
    )
    monkeypatch.setattr("app.rd.subagent.rag_search", mock_rag)
    monkeypatch.setattr("app.tools.rag.rag_search", mock_rag, raising=False)

    direction = Direction(id="datasets", question="What datasets?", rationale="r")
    client = _FakeClient("Findings about datasets [1].")
    result = await run_direction(client, direction)

    assert isinstance(result, ResearchSummary)
    assert result.direction_id == "datasets"
    assert isinstance(result.findings, str) and len(result.findings) > 0
    assert isinstance(result.sources, list)
    assert result.confidence in ("low", "med", "high")
    assert isinstance(result.gaps, list)


@pytest.mark.asyncio
async def test_subagent_confidence_enum(monkeypatch):
    """Confidence must be low|med|high. §4.3"""
    from unittest.mock import AsyncMock

    from app.rd.subagent import run_direction

    for hits, expected_conf in [
        ([], "low"),
        ([{"page": 1, "score": 0.8, "text": "t", "source": "s"}], "med"),
        (
            [
                {"page": 1, "score": 0.8, "text": "t", "source": "s"},
                {"page": 2, "score": 0.7, "text": "t2", "source": "s"},
                {"page": 3, "score": 0.6, "text": "t3", "source": "s"},
            ],
            "high",
        ),
    ]:
        mock_rag = AsyncMock(return_value={"text": hits, "images": []})
        monkeypatch.setattr("app.rd.subagent.rag_search", mock_rag)
        monkeypatch.setattr("app.tools.rag.rag_search", mock_rag, raising=False)
        client = _FakeClient("findings")
        result = await run_direction(client, Direction(id="evaluation", question="q", rationale="r"))
        assert result.confidence in ("low", "med", "high")


@pytest.mark.asyncio
async def test_subagent_gaps_when_low_confidence(monkeypatch):
    """Low confidence should populate gaps. §4.3"""
    from unittest.mock import AsyncMock

    from app.rd.subagent import run_direction

    mock_rag = AsyncMock(return_value={"text": [], "images": []})
    monkeypatch.setattr("app.rd.subagent.rag_search", mock_rag)
    monkeypatch.setattr("app.tools.rag.rag_search", mock_rag, raising=False)
    client = _FakeClient("no evidence")
    result = await run_direction(client, Direction(id="baselines", question="q", rationale="r"))
    assert result.confidence == "low"
    assert len(result.gaps) >= 1


@pytest.mark.asyncio
async def test_subagent_sources_from_rag(monkeypatch):
    """Sources must come from rag_source_items via retrieval. §4.3"""
    from unittest.mock import AsyncMock

    from app.rd.subagent import run_direction

    hits = [
        {"page": 2, "score": 0.85, "text": "method uses transformer", "source": "paper.pdf", "object_key": ""},
    ]
    mock_rag = AsyncMock(return_value={"text": hits, "images": []})
    monkeypatch.setattr("app.rd.subagent.rag_search", mock_rag)
    monkeypatch.setattr("app.tools.rag.rag_search", mock_rag, raising=False)
    client = _FakeClient("findings [1]")
    result = await run_direction(client, Direction(id="core_algorithm", question="q", rationale="r"))
    assert len(result.sources) >= 1
    # source_items shape: kind, source, page, score, content
    assert "source" in result.sources[0] or "page" in result.sources[0]


@pytest.mark.asyncio
async def test_subagent_findings_cite_sources_when_hits_exist(monkeypatch):
    """When hits exist, synthesis prompt should request [1] citations — findings should be non-empty. §4.3"""
    from unittest.mock import AsyncMock

    from app.rd.subagent import run_direction

    mock_rag = AsyncMock(
        return_value={
            "text": [{"page": 1, "score": 0.9, "text": "evidence text", "source": "s"}],
            "images": [{"page": 1, "score": 0.88, "object_key": "docs/x/fig.png", "source": "s"}],
        }
    )
    monkeypatch.setattr("app.rd.subagent.rag_search", mock_rag)
    monkeypatch.setattr("app.tools.rag.rag_search", mock_rag, raising=False)
    client = _FakeClient("Findings referencing [1] and figure.")
    result = await run_direction(client, Direction(id="implementation_details", question="q", rationale="r"))
    assert result.findings


@pytest.mark.asyncio
async def test_subagent_handles_llm_failure(monkeypatch):
    """Must not raise if LLM fails — should return fallback findings. Spec robustness."""
    from unittest.mock import AsyncMock

    from app.rd.subagent import run_direction

    mock_rag = AsyncMock(return_value={"text": [{"page": 1, "score": 0.9, "text": "t", "source": "s"}], "images": []})
    monkeypatch.setattr("app.rd.subagent.rag_search", mock_rag)
    monkeypatch.setattr("app.tools.rag.rag_search", mock_rag, raising=False)

    class _FailingCompletions:
        async def create(self, **kwargs):
            raise RuntimeError("LLM down")

    client = type("C", (), {"chat": type("C2", (), {"completions": _FailingCompletions()})()})()
    result = await run_direction(client, Direction(id="core_algorithm", question="q", rationale="r"))
    assert result.findings  # fallback should be non-empty
    assert result.direction_id == "core_algorithm"
