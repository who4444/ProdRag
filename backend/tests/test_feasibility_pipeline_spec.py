"""Feasibility pipeline: event order, clarify handshake, emit-with-risks (M2)."""

import json

import pytest

from app.feasibility.schemas import AnalyzeRequest


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "core.py").write_text("class Engine:\n    pass\n\n\ndef run():\n    pass\n", encoding="utf-8")
    return repo


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
    """Routes by system prompt so all pipeline stages run against one fake."""

    def __init__(self, overrides: dict | None = None):
        self.overrides = overrides or {}
        self.systems: list[str] = []
        self.chat = type("C", (), {"completions": type("CC", (), {"create": self._create})()})()

    async def _create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        self.systems.append(system[:40])
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


@pytest.fixture
def patched(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock

    monkeypatch.setattr("app.feasibility.subagent.code_search", AsyncMock(return_value={"hits": _hits()}))
    monkeypatch.setattr("app.feasibility.pipeline.db.feasibility_run_create", AsyncMock())
    monkeypatch.setattr("app.feasibility.pipeline.db.feasibility_run_update", AsyncMock())
    monkeypatch.setattr("app.feasibility.pipeline.db.feasibility_report_upsert", AsyncMock())
    return _repo(tmp_path)


async def _collect(client, req, root):
    from app.feasibility.pipeline import feasibility_analyze

    return [evt async for evt in feasibility_analyze(client, req, root)]


@pytest.mark.asyncio
async def test_pipeline_emits_start_map_claims_report(patched):
    client = _RoutingClient()
    events = await _collect(client, AnalyzeRequest(idea="add caching", repo_id="r1"), patched)

    kinds = [e["event"] for e in events]
    assert kinds[0] == "start"
    assert kinds[1] == "understand"
    assert "map" in kinds
    assert kinds.count("claim") == 3
    assert kinds[-1] == "report"

    report = events[-1]["data"]
    assert report["verdict"] == "plausible"
    assert report["repo_id"] == "r1"
    assert len(report["claims"]) == 3
    assert all(c["status"] == "supported" for c in report["claims"])
    assert report["usage"]["calls"] >= 4  # understand + enrich + 3 subagents + judge
    assert report["codebase_map"]["extension_points"][0]["path"] == "pkg/core.py"


@pytest.mark.asyncio
async def test_pipeline_clarify_stops_before_map(patched):
    clarify = {"ready": False, "understanding": {"interpretation": "vague"}, "questions": ["Which module?"]}
    client = _RoutingClient({"understand": clarify})
    events = await _collect(client, AnalyzeRequest(idea="make it better", repo_id="r1"), patched)

    assert [e["event"] for e in events] == ["start", "clarify"]
    assert events[-1]["questions"] == ["Which module?"]


@pytest.mark.asyncio
async def test_pipeline_emits_even_when_db_persist_fails(patched, monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        "app.feasibility.pipeline.db.feasibility_run_create", AsyncMock(side_effect=RuntimeError("db down"))
    )
    monkeypatch.setattr(
        "app.feasibility.pipeline.db.feasibility_report_upsert", AsyncMock(side_effect=RuntimeError("db down"))
    )
    events = await _collect(_RoutingClient(), AnalyzeRequest(idea="add caching", repo_id="r1"), patched)
    assert any(e["event"] == "report" for e in events)  # emit-with-risks, never blocks


@pytest.mark.asyncio
async def test_repo_root_rejects_unknown_repo(monkeypatch):
    from unittest.mock import AsyncMock

    from app.feasibility.pipeline import repo_root

    monkeypatch.setattr("app.feasibility.pipeline.db.repo_get", AsyncMock(return_value=None))
    with pytest.raises(ValueError):
        await repo_root("nope")
