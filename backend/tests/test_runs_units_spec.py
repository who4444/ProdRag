"""Run units: report->artifact bridge, PoC evaluation, usage-tracking client (M3)."""

import json

import pytest


def _report(verdict="plausible"):
    return {
        "idea": "add caching",
        "understanding": {
            "interpretation": "add caching to the engine",
            "problem": "slow",
            "expected_outcome": "faster",
            "assumptions": [],
            "hypotheses": ["cache hit rate improves"],
            "ambiguities": [],
        },
        "claims": [
            {
                "id": "technical-1",
                "dimension": "technical",
                "statement": "Engine owns the state to cache.",
                "evidence": [{"path": "pkg/core.py", "start_line": 1, "end_line": 2, "valid": True}],
                "confidence": "high",
                "status": "supported",
            },
            {
                "id": "data-1",
                "dimension": "data",
                "statement": "No labeled data available.",
                "evidence": [],
                "confidence": "low",
                "status": "assumption",
            },
        ],
        "dimensions": {
            "technical": {"plausible": True, "rationale": "ok", "blockers": [], "missing_evidence": []},
            "integration": {"plausible": False, "rationale": "no", "blockers": ["breaks API"], "missing_evidence": []},
            "data": {"plausible": False, "rationale": "no data", "blockers": ["no dataset"], "missing_evidence": []},
        },
        "verdict": verdict,
        "blockers": ["breaks API", "no dataset"],
        "alternatives": ["cache only embeds"],
        "reframed_idea": "cache only the embedding call",
        "codebase_map": {
            "extension_points": [
                {"path": "pkg/core.py", "symbol": "Engine", "reason": "state owner", "evidence": []}
            ],
            "data_available": [],
            "data_missing": ["labeled set"],
            "data_cleanliness": "unknown",
        },
    }


def test_bridge_carries_evidence_and_hypotheses_into_artifact():
    from app.runs.bridge import report_to_artifact

    artifact = report_to_artifact(_report(), "add caching")
    dims = {s.direction_id: s for s in artifact.summaries}

    assert artifact.validation.feasible is True
    assert artifact.validation.coverage["technical"] is True
    assert artifact.validation.coverage["integration"] is False
    assert artifact.validation.risks == ["breaks API", "no dataset"]
    assert "pkg/core.py:1-2" in dims["technical"].findings
    assert "PoC hypothesis: cache hit rate improves" in artifact.demo_spec.acceptance_criteria
    assert any(c.startswith("demo.py exits 0") for c in artifact.demo_spec.acceptance_criteria)
    assert any("JSON" in c for c in artifact.demo_spec.acceptance_criteria)  # explicit I/O contract
    assert "pkg/core.py::Engine" in artifact.demo_spec.pseudocode
    assert artifact.requirements.goal == "add caching"


def test_bridge_uses_poc_plan_when_present():
    from app.runs.bridge import report_to_artifact

    plan = {
        "chosen_method": "wrap rag_search with an LRU cache",
        "implementation_requirements": ["cache in tools/rag.py", "count embedding calls"],
        "evaluation_metrics": [
            {"name": "embedding_calls", "direction": "bounded", "threshold": "<=1 per unique question"}
        ],
        "system_context": "backend/app/tools/rag.py::rag_search",
    }
    artifact = report_to_artifact(_report(), "add caching", plan)

    assert "wrap rag_search with an LRU cache" in artifact.demo_spec.core_algorithm
    assert "cache in tools/rag.py" in artifact.demo_spec.pseudocode
    assert "System context: backend/app/tools/rag.py::rag_search" in artifact.demo_spec.pseudocode
    assert any("embedding_calls" in c and "bounded" in c for c in artifact.demo_spec.acceptance_criteria)
    # evidence still rides along
    assert "Evidence:" in artifact.demo_spec.core_algorithm or "Engine owns the state" in artifact.demo_spec.core_algorithm


class _FakeClient:
    def __init__(self, payload):
        async def _create(**kwargs):
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


class _Usage:
    prompt_tokens = 7
    completion_tokens = 3


class _PlainClient:
    def __init__(self):
        async def _create(**kwargs):
            return type(
                "R",
                (),
                {"choices": [type("C", (), {"message": type("M", (), {"content": "content"})()})()], "usage": _Usage()},
            )()

        self.chat = type("C", (), {"completions": type("CC", (), {"create": staticmethod(_create)})()})()


SANDBOX_OK = {"passed": True, "pytest_log": "1 passed", "demo_log": "ok", "returncode": 0, "runtime_ms": 42}
SANDBOX_FAIL = {"passed": False, "pytest_log": "1 failed", "demo_log": "boom", "returncode": 1, "runtime_ms": 42}
UNDERSTANDING = {"hypotheses": ["cache hit rate improves"]}


@pytest.mark.asyncio
async def test_evaluate_poc_pass_with_supported_hypothesis():
    from app.runs.evaluate import evaluate_poc

    client = _FakeClient(
        {"hypotheses": [{"hypothesis": "cache hit rate improves", "supported": True, "evidence": "1 passed"}], "caveats": [], "verdict": "pass"}
    )
    report = await evaluate_poc(client, UNDERSTANDING, SANDBOX_OK, {"calls": 4, "cost_usd": 0.01}, wall_ms=100)
    assert report.runnable is True
    assert report.spec_fulfilled is True
    assert report.verdict == "pass"
    assert report.llm_calls == 4
    assert report.runtime_ms == 42


def test_extract_demo_metrics_finds_last_json_object():
    from app.runs.evaluate import _extract_demo_metrics

    text = "preamble\n{\"old\": 1}\nmore text\n{\"cache_hits\": 2, \"embedding_calls\": 1}\n"
    assert _extract_demo_metrics(text) == {"cache_hits": 2, "embedding_calls": 1}
    assert _extract_demo_metrics("no json here") == {}
    assert _extract_demo_metrics("") == {}


@pytest.mark.asyncio
async def test_evaluate_poc_surfaces_performance_and_worthiness():
    from app.runs.evaluate import evaluate_poc

    sandbox = {
        **SANDBOX_OK,
        "demo_runtime_ms": 12,
        "memory_peak_kb": 40960,
        "demo_log": '{"cache_hits": 2, "embedding_calls": 1}',
    }
    client = _FakeClient(
        {
            "hypotheses": [{"hypothesis": "cache hit rate improves", "supported": True, "evidence": "1 passed"}],
            "caveats": [],
            "verdict": "pass",
            "performance_summary": "cache removes repeated embedding work",
            "technical_debt": ["in-process cache only"],
            "extensibility": ["needs eviction metrics for production"],
            "worthiness": "marginal",
        }
    )
    report = await evaluate_poc(client, UNDERSTANDING, sandbox, {"calls": 5, "cost_usd": 0.02}, wall_ms=100)

    assert report.performance.demo_runtime_ms == 12
    assert report.performance.memory_peak_kb == 40960
    assert report.performance.metrics == {"cache_hits": 2, "embedding_calls": 1}
    assert report.worthiness.worthiness == "marginal"
    assert report.worthiness.technical_debt == ["in-process cache only"]
    assert report.worthiness.cost_usd == 0.02


@pytest.mark.asyncio
async def test_evaluate_poc_worthiness_fallback_when_judge_silent():
    from app.runs.evaluate import evaluate_poc

    client = _FakeClient(
        {
            "hypotheses": [{"hypothesis": "cache hit rate improves", "supported": True, "evidence": "1 passed"}],
            "caveats": [],
            "verdict": "pass",
        }
    )
    report = await evaluate_poc(client, UNDERSTANDING, SANDBOX_OK, {"calls": 1}, wall_ms=10)
    assert report.worthiness.worthiness == "worth_it"

    failed = await evaluate_poc(_FailingClient(), UNDERSTANDING, SANDBOX_FAIL, {}, wall_ms=10)
    assert failed.worthiness.worthiness == "not_worth_it"


@pytest.mark.asyncio
async def test_evaluate_poc_maps_reworded_hypotheses_by_position():
    """The judge often prefixes/reworks items; results must still map to inputs."""
    from app.runs.evaluate import evaluate_poc

    client = _FakeClient(
        {
            "hypotheses": [
                {"hypothesis": "H1: cache hit rate improves", "supported": True, "evidence": "1 passed"},
            ],
            "caveats": [],
            "verdict": "pass",
        }
    )
    report = await evaluate_poc(client, UNDERSTANDING, SANDBOX_OK, {"calls": 1}, wall_ms=10)
    assert report.hypotheses[0].hypothesis == "cache hit rate improves"  # original text preserved
    assert report.hypotheses[0].supported is True
    assert report.hypotheses[0].evidence == "1 passed"


@pytest.mark.asyncio
async def test_evaluate_poc_judge_failure_degrades_to_not_evaluated():
    from app.runs.evaluate import evaluate_poc

    report = await evaluate_poc(_FailingClient(), UNDERSTANDING, SANDBOX_OK, {"calls": 1}, wall_ms=10)
    assert report.verdict == "pass_with_caveats"
    assert report.hypotheses[0].supported is False
    assert report.hypotheses[0].evidence == "not evaluated"


@pytest.mark.asyncio
async def test_evaluate_poc_sandbox_failure_is_fail():
    from app.runs.evaluate import evaluate_poc

    report = await evaluate_poc(_FailingClient(), UNDERSTANDING, SANDBOX_FAIL, {}, wall_ms=10)
    assert report.runnable is False
    assert report.verdict == "fail"
    assert any("sandbox failed" in c for c in report.caveats)


@pytest.mark.asyncio
async def test_usage_tracking_client_records_into_run_ledger():
    from app.core import llm
    from app.runs.instrument import UsageTrackingClient

    usage = llm.activate_usage("test-run-1")
    client = UsageTrackingClient(_PlainClient())
    resp = await client.chat.completions.create(model="x", messages=[{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content == "content"
    assert usage.calls == 1
    assert usage.prompt_tokens == 7 and usage.completion_tokens == 3
    llm.release_usage("test-run-1")
