"""Dimension subagent: retrieval -> cited claims -> citation gate (M2)."""

import json

import pytest

from app.feasibility.planner import build_dimensions
from app.feasibility.schemas import IdeaUnderstanding


def _repo(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "core.py").write_text("class Engine:\n    pass\n\n\ndef run():\n    pass\n", encoding="utf-8")
    return tmp_path


def _hits():
    return [
        {
            "path": "pkg/core.py",
            "symbol": "Engine",
            "kind": "class",
            "start_line": 1,
            "end_line": 2,
            "text": "class Engine:\n    pass",
            "score": 0.9,
        }
    ]


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


@pytest.mark.asyncio
async def test_assess_dimension_gates_claims(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from app.feasibility.subagent import assess_dimension

    monkeypatch.setattr("app.feasibility.subagent.code_search", AsyncMock(return_value={"hits": _hits()}))
    payload = {
        "claims": [
            {
                "statement": "Engine already owns the state we need to cache.",
                "confidence": "high",
                "evidence": [{"path": "pkg/core.py", "start_line": 1, "end_line": 2}],
            },
            {
                "statement": "A cache layer is already wired into run().",
                "confidence": "low",
                "evidence": [{"path": "pkg/ghost.py", "start_line": 1, "end_line": 2}],
            },
        ]
    }
    dim = build_dimensions(IdeaUnderstanding(interpretation="add caching"), "add caching")[0]

    claims, hits = await assess_dimension(
        _FakeClient(payload), dim, "add caching", IdeaUnderstanding(interpretation="add caching"), cmap=_fake_cmap(), root=_repo(tmp_path), repo_id="r1"
    )

    assert [c.status for c in claims] == ["supported", "unsupported"]
    assert claims[0].evidence[0].valid is True
    assert claims[1].evidence[0].reason == "file not found"
    assert hits == _hits()
    assert len(claims) == 2


@pytest.mark.asyncio
async def test_assess_dimension_llm_failure_yields_provisional_claim(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from app.feasibility.subagent import assess_dimension

    monkeypatch.setattr("app.feasibility.subagent.code_search", AsyncMock(return_value={"hits": _hits()}))
    dim = build_dimensions(IdeaUnderstanding(), "idea")[0]

    claims, _ = await assess_dimension(
        _FailingClient(), dim, "idea", IdeaUnderstanding(), cmap=_fake_cmap(), root=_repo(tmp_path), repo_id="r1"
    )

    assert len(claims) == 1
    assert claims[0].provisional is True
    assert claims[0].status == "unsupported"
    assert claims[0].evidence[0].valid is True  # raw retrieval attached for manual review


@pytest.mark.asyncio
async def test_assess_dimension_search_failure_still_returns_claim(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from app.feasibility.subagent import assess_dimension

    monkeypatch.setattr("app.feasibility.subagent.code_search", AsyncMock(side_effect=RuntimeError("qdrant down")))
    dim = build_dimensions(IdeaUnderstanding(), "idea")[0]

    claims, hits = await assess_dimension(
        _FailingClient(), dim, "idea", IdeaUnderstanding(), cmap=_fake_cmap(), root=_repo(tmp_path), repo_id="r1"
    )
    assert hits == []
    assert claims[0].status == "unsupported"


def _fake_cmap():
    from app.feasibility.schemas import CodebaseMap

    return CodebaseMap(impacted_modules=["pkg/core.py"])
