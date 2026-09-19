"""Judge: evidence gate + verdict mapping + emit-with-risks fallback (M2)."""

import json

import pytest

from app.feasibility.judge import FALLBACK_ALTERNATIVES, apply_evidence_gate, judge_feasibility
from app.feasibility.schemas import DIMENSION_IDS, CodebaseMap, EvidenceRef, FeasibilityClaim, IdeaUnderstanding


def _claim(dimension: str, status: str) -> FeasibilityClaim:
    return FeasibilityClaim(
        id=f"{dimension}-1",
        dimension=dimension,
        statement=f"{dimension} claim",
        evidence=[EvidenceRef(path="pkg/core.py", valid=True)],
        status=status,
    )


def _dims(plausible: bool = True) -> dict:
    from app.feasibility.schemas import DimensionVerdict

    return {d: DimensionVerdict(plausible=plausible, rationale="x") for d in DIMENSION_IDS}


def test_gate_blocks_plausible_without_supported_claims():
    dimensions = _dims(plausible=True)
    claims = [
        _claim("technical", "supported"),
        _claim("integration", "assumption"),
        _claim("data", "supported"),
    ]

    dims, verdict, blockers = apply_evidence_gate(claims, dimensions, "plausible")

    assert dims["technical"].plausible is True
    assert dims["integration"].plausible is False
    assert dims["data"].plausible is True
    assert verdict == "plausible_with_risks"  # exactly 1 implausible
    assert blockers and any("integration" in b for b in blockers)


def test_gate_two_implausible_is_infeasible():
    dimensions = _dims(plausible=True)
    claims = [_claim("technical", "supported")]
    dims, verdict, blockers = apply_evidence_gate(claims, dimensions, "plausible")
    assert verdict == "infeasible"
    assert len(blockers) >= 2


def test_gate_all_supported_maps_to_plausible_or_with_risks():
    claims = [_claim(d, "supported") for d in DIMENSION_IDS]

    _, verdict, _ = apply_evidence_gate(claims, _dims(plausible=True), "plausible")
    assert verdict == "plausible"

    _, verdict, _ = apply_evidence_gate(claims, _dims(plausible=True), "plausible_with_risks")
    assert verdict == "plausible_with_risks"

    _, verdict, _ = apply_evidence_gate(claims, _dims(plausible=True), "plausible_with_risks")
    assert verdict == "plausible_with_risks"


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


@pytest.mark.asyncio
async def test_judge_llm_output_is_gated():
    payload = {
        "dimensions": {d: {"plausible": True, "rationale": "looks fine", "blockers": [], "missing_evidence": []} for d in DIMENSION_IDS},
        "verdict": "plausible",
        "blockers": [],
        "alternatives": [],
        "reframed_idea": None,
    }
    claims = [_claim("technical", "supported")]  # only technical has evidence

    result = await judge_feasibility(_FakeClient(payload), "idea", IdeaUnderstanding(), CodebaseMap(), claims)

    assert result["dimensions"]["integration"]["plausible"] is False
    assert result["verdict"] == "infeasible"
    assert result["alternatives"]  # fallback suggestions injected


@pytest.mark.asyncio
async def test_judge_llm_failure_uses_deterministic_verdicts():
    claims = [_claim("technical", "supported"), _claim("data", "unsupported")]
    result = await judge_feasibility(_FailingClient(), "idea", IdeaUnderstanding(), CodebaseMap(), claims)

    assert result["dimensions"]["technical"]["plausible"] is True
    assert result["dimensions"]["data"]["plausible"] is False
    assert result["verdict"] == "infeasible"  # integration + data unsupported
    assert result["alternatives"] == FALLBACK_ALTERNATIVES
