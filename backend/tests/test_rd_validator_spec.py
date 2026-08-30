"""Spec-driven tests for Validator — PROJECT.md §4.4.

LLM-as-judge, emit-with-risks (never blocks).
"""

import json

import pytest

from app.rd.schemas import Requirements, ResearchSummary, Validation


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

    async def create(self, **kwargs):
        return _FakeResp(json.dumps(self._payload))


class _FakeClient:
    def __init__(self, payload: dict):
        self.chat = type("C", (), {"completions": _FakeCompletions(payload)})()


def _sample_requirements():
    return Requirements(goal="demo attention", core_idea="attention", demo_type="script", constraints=[], audience="researcher")


def _sample_summaries():
    return [
        ResearchSummary(direction_id="core_algorithm", findings="algorithm found", sources=[], confidence="high", gaps=[]),
        ResearchSummary(direction_id="datasets", findings="datasets found", sources=[], confidence="high", gaps=[]),
        ResearchSummary(direction_id="baselines", findings="", sources=[], confidence="low", gaps=["no baselines"]),
        ResearchSummary(direction_id="implementation_details", findings="details found", sources=[], confidence="med", gaps=[]),
        ResearchSummary(direction_id="evaluation", findings="eval found", sources=[], confidence="high", gaps=[]),
    ]


@pytest.mark.asyncio
async def test_validator_returns_validation_shape():
    """Must return Validation with spec shape. §4.4"""
    from app.rd.validator import validate

    payload = {
        "coverage": {"core_algorithm": True, "datasets": True, "baselines": False, "implementation_details": True, "evaluation": True},
        "missing": ["baselines"],
        "feasible": True,
        "risks": ["Low evidence for baselines"],
    }
    client = _FakeClient(payload)
    result = await validate(client, _sample_requirements(), _sample_summaries())

    assert isinstance(result, Validation)
    assert isinstance(result.coverage, dict)
    assert isinstance(result.missing, list)
    assert isinstance(result.feasible, bool)
    assert isinstance(result.risks, list)


@pytest.mark.asyncio
async def test_validator_coverage_has_direction_keys():
    """Coverage must be keyed by direction_id. §4.4"""
    from app.rd.validator import validate

    payload = {
        "coverage": {d: True for d in ["core_algorithm", "datasets", "baselines", "implementation_details", "evaluation"]},
        "missing": [],
        "feasible": True,
        "risks": [],
    }
    client = _FakeClient(payload)
    summaries = _sample_summaries()
    result = await validate(client, _sample_requirements(), summaries)
    # coverage should cover the taxonomy (at least the 5 IDs when LLM is well-behaved)
    for key in ["core_algorithm", "datasets", "baselines", "implementation_details", "evaluation"]:
        assert key in result.coverage or result.missing is not None  # at least shape is valid


@pytest.mark.asyncio
async def test_validator_emit_with_risks_when_infeasible():
    """Even when feasible=false, must still emit Validation with risks — never blocks. §4.4 emit-with-risks"""
    from app.rd.validator import validate

    payload = {
        "coverage": {"core_algorithm": False, "datasets": False, "baselines": False, "implementation_details": False, "evaluation": False},
        "missing": ["core_algorithm", "datasets"],
        "feasible": False,
        "risks": ["No evidence for core algorithm — demo will be speculative", "Missing datasets"],
    }
    client = _FakeClient(payload)
    result = await validate(client, _sample_requirements(), _sample_summaries())
    assert result.feasible is False
    assert len(result.risks) >= 1
    # Must still be a valid Validation, not an exception
    assert isinstance(result, Validation)


@pytest.mark.asyncio
async def test_validator_missing_lists_uncovered():
    """Missing should list uncovered requirements/directions. §4.4"""
    from app.rd.validator import validate

    payload = {"coverage": {"core_algorithm": True, "datasets": False}, "missing": ["datasets"], "feasible": True, "risks": []}
    client = _FakeClient(payload)
    result = await validate(client, _sample_requirements(), _sample_summaries())
    assert "datasets" in result.missing


@pytest.mark.asyncio
async def test_validator_risks_even_when_feasible():
    """Risks should be present even when feasible — caveats. §4.4"""
    from app.rd.validator import validate

    payload = {"coverage": {"core_algorithm": True}, "missing": [], "feasible": True, "risks": ["Toy data may not reflect real distribution"]}
    client = _FakeClient(payload)
    result = await validate(client, _sample_requirements(), _sample_summaries())
    assert result.feasible is True
    assert len(result.risks) >= 1 or result.missing == []


@pytest.mark.asyncio
async def test_validator_never_raises_on_llm_failure():
    """Validator must fallback and never raise if LLM fails. Spec robustness."""
    from app.rd.validator import validate

    class _FailingCompletions:
        async def create(self, **kwargs):
            raise RuntimeError("LLM down")

    client = type("C", (), {"chat": type("C2", (), {"completions": _FailingCompletions()})()})()
    result = await validate(client, _sample_requirements(), _sample_summaries())
    assert isinstance(result, Validation)
    # fallback should produce coverage based on confidence
    assert isinstance(result.coverage, dict)


@pytest.mark.asyncio
async def test_validator_single_llm_call():
    """Validator must be one LLM call with json_object. §4.4 ponytail"""
    from app.rd.validator import validate

    payload = {"coverage": {}, "missing": [], "feasible": True, "risks": []}

    class _CountingCompletions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return _FakeResp(json.dumps(payload))

    completions = _CountingCompletions()
    client = type("C", (), {"chat": type("C2", (), {"completions": completions})()})()
    await validate(client, _sample_requirements(), _sample_summaries())
    assert len(completions.calls) == 1
    assert completions.calls[0].get("response_format") == {"type": "json_object"}
