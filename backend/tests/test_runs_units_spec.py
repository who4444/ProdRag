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
    assert "demo.py exits 0" in artifact.demo_spec.acceptance_criteria
    assert "pkg/core.py::Engine" in artifact.demo_spec.pseudocode
    assert artifact.requirements.goal == "add caching"


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
