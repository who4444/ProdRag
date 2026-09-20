"""PoC plan (lean research agent): LLM path, fallback, optional KB (M3.5)."""

import json

import pytest


def _report() -> dict:
    return {
        "claims": [
            {
                "dimension": "technical",
                "status": "supported",
                "statement": "rag_search is the shared retrieval entry point.",
                "evidence": [{"path": "backend/app/tools/rag.py", "start_line": 11, "end_line": 14, "valid": True}],
            }
        ],
        "blockers": ["no session_id on /query"],
        "codebase_map": {"extension_points": [{"path": "backend/app/tools/rag.py", "symbol": "rag_search"}]},
    }


def _understanding() -> dict:
    return {
        "interpretation": "add an LRU cache to rag_search",
        "hypotheses": ["repeated questions perform zero embedding calls"],
    }


def _cmap() -> dict:
    return {
        "extension_points": [{"path": "backend/app/tools/rag.py", "symbol": "rag_search", "reason": "entry point"}],
        "impacted_modules": ["backend/app/tools/rag.py", "backend/app/rag/retrieval.py"],
        "data_available": [],
        "data_missing": [],
    }


class _FakeClient:
    def __init__(self, payload):
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


@pytest.fixture
def no_kb(monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setattr("app.tools.rag.rag_search", AsyncMock(return_value={"text": [], "images": []}))


@pytest.mark.asyncio
async def test_build_poc_plan_parses_and_sanitizes(monkeypatch, no_kb):
    from app.feasibility.poc_plan import build_poc_plan

    payload = {
        "chosen_method": "wrap rag_search with an LRU cache",
        "implementation_requirements": ["cache in tools/rag.py", "count embedding calls"],
        "experiment_objective": "repeated questions cost one embedding call",
        "evaluation_metrics": [
            {"name": "embedding_calls", "direction": "bounded", "threshold": "<=1 per unique question"},
            {"name": "", "direction": "increase", "threshold": "ignored"},
            {"name": "cache_hits", "direction": "nonsense", "threshold": ">=1"},
        ],
        "system_context": "backend/app/tools/rag.py::rag_search",
    }
    plan = await build_poc_plan(_FakeClient(payload), "add caching", _understanding(), _report(), _cmap())

    assert plan.chosen_method.startswith("wrap rag_search")
    assert plan.implementation_requirements[0] == "cache in tools/rag.py"
    assert [m.name for m in plan.evaluation_metrics] == ["embedding_calls", "cache_hits"]
    assert plan.evaluation_metrics[1].direction == "present"  # invalid value coerced
    assert plan.system_context == "backend/app/tools/rag.py::rag_search"


@pytest.mark.asyncio
async def test_build_poc_plan_falls_back_on_llm_failure(no_kb):
    from app.feasibility.poc_plan import build_poc_plan

    plan = await build_poc_plan(_FailingClient(), "add caching", _understanding(), _report(), _cmap())

    assert "LRU cache" in plan.chosen_method
    assert any("rag_search" in r for r in plan.implementation_requirements)
    assert plan.experiment_objective == "repeated questions perform zero embedding calls"
    assert plan.evaluation_metrics and "repeated questions" in plan.evaluation_metrics[0].threshold
    assert "backend/app/tools/rag.py" in plan.system_context


@pytest.mark.asyncio
async def test_build_poc_plan_survives_kb_failure(monkeypatch):
    from unittest.mock import AsyncMock

    from app.feasibility.poc_plan import build_poc_plan

    monkeypatch.setattr("app.tools.rag.rag_search", AsyncMock(side_effect=RuntimeError("KB down")))
    client = _FakeClient(
        {
            "chosen_method": "m",
            "implementation_requirements": ["r"],
            "experiment_objective": "o",
            "evaluation_metrics": [{"name": "x", "direction": "present", "threshold": "y"}],
            "system_context": "s",
        }
    )
    plan = await build_poc_plan(client, "idea", _understanding(), _report(), _cmap())
    assert plan.chosen_method == "m"
    assert len(client.calls) == 1  # KB failure did not block the LLM call


@pytest.mark.asyncio
async def test_build_poc_plan_includes_kb_hits_when_present(monkeypatch):
    from unittest.mock import AsyncMock

    from app.feasibility.poc_plan import build_poc_plan

    monkeypatch.setattr(
        "app.tools.rag.rag_search",
        AsyncMock(return_value={"text": [{"source": "paper.pdf", "page": 3, "text": "prior art on caching"}], "images": []}),
    )
    client = _FakeClient(
        {
            "chosen_method": "m",
            "implementation_requirements": ["r"],
            "experiment_objective": "o",
            "evaluation_metrics": [{"name": "x", "direction": "present", "threshold": "y"}],
            "system_context": "s",
        }
    )
    await build_poc_plan(client, "idea", _understanding(), _report(), _cmap())
    user_message = client.calls[0]["messages"][-1]["content"]
    assert "Knowledge base hits" in user_message
    assert "prior art on caching" in user_message
