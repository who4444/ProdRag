"""Spec-driven tests for Research Artifact — PROJECT.md §4.5.

DemoSpec + ResearchArtifact handoff to coding module.
"""

import pytest

from app.rd.schemas import DemoSpec, Requirements, ResearchArtifact, ResearchSummary, Validation


def _req():
    return Requirements(goal="demo attention", core_idea="attention is all you need", demo_type="script", constraints=[], audience="researcher")


def _summaries_with_dup_sources():
    return [
        ResearchSummary(
            direction_id="core_algorithm",
            findings="Self-attention steps: QKV projection, scaled dot-product, softmax, weighted sum.",
            sources=[{"source": "paper.pdf", "page": 1, "content": "attention mechanism", "score": 0.9}],
            confidence="high",
        ),
        ResearchSummary(
            direction_id="datasets",
            findings="Toy data: random sequences length 10.",
            sources=[{"source": "paper.pdf", "page": 1, "content": "attention mechanism", "score": 0.9}],  # duplicate
            confidence="high",
        ),
    ]


def test_artifact_has_required_fields():
    """ResearchArtifact must contain requirements, summaries, validation, demo_spec, sources, created_at. §4.5"""
    from app.rd.artifacts import build_artifact

    req = _req()
    summaries = _summaries_with_dup_sources()
    validation = Validation(coverage={"core_algorithm": True}, feasible=True, risks=[])
    artifact = build_artifact(req, summaries, validation)

    assert isinstance(artifact, ResearchArtifact)
    assert artifact.requirements == req
    assert artifact.summaries == summaries
    assert artifact.validation == validation
    assert isinstance(artifact.demo_spec, DemoSpec)
    assert isinstance(artifact.sources, list)
    assert isinstance(artifact.created_at, float) and artifact.created_at > 0


def test_artifact_demo_spec_entrypoint():
    """DemoSpec.entrypoint must be demo.py. §4.5"""
    from app.rd.artifacts import build_artifact

    artifact = build_artifact(_req(), _summaries_with_dup_sources(), Validation(feasible=True))
    assert artifact.demo_spec.entrypoint == "demo.py"


def test_artifact_tech_stack_contains_python():
    """tech_stack must contain python and baseline deps. §4.5"""
    from app.rd.artifacts import build_artifact

    artifact = build_artifact(_req(), _summaries_with_dup_sources(), Validation(feasible=True))
    assert "python" in artifact.demo_spec.tech_stack
    assert len(artifact.demo_spec.tech_stack) >= 1


def test_artifact_acceptance_criteria_non_empty():
    """acceptance_criteria are the contract — must be non-empty and include runnable checks. §4.5"""
    from app.rd.artifacts import build_artifact

    artifact = build_artifact(_req(), _summaries_with_dup_sources(), Validation(feasible=True))
    criteria = artifact.demo_spec.acceptance_criteria
    assert len(criteria) >= 1
    # must include the canonical runnable check
    assert any("demo.py" in c and "0" in c for c in criteria), "must include 'demo.py exits 0' style check"
    # at least one criterion should be a toy metric/behavior check
    assert any(len(c) > 10 for c in criteria)


def test_artifact_sources_deduped():
    """Sources must be deduped by source+page+content across summaries. §4.5"""
    from app.rd.artifacts import build_artifact, dedupe_sources

    summaries = _summaries_with_dup_sources()
    # direct deduper
    assert len(dedupe_sources(summaries)) == 1, "duplicate source+page+content must be collapsed"

    artifact = build_artifact(_req(), summaries, Validation(feasible=True))
    assert len(artifact.sources) == 1


def test_artifact_core_algorithm_populated():
    """demo_spec.core_algorithm and pseudocode must be populated from core_algorithm summary or requirements. §4.5"""
    from app.rd.artifacts import build_artifact

    summaries = _summaries_with_dup_sources()
    artifact = build_artifact(_req(), summaries, Validation(feasible=True))
    assert artifact.demo_spec.core_algorithm
    assert artifact.demo_spec.pseudocode


def test_artifact_validation_risks_propagated():
    """When validation has risks, acceptance_criteria should acknowledge them. §4.5/§4.4 emit-with-risks"""
    from app.rd.artifacts import build_artifact

    req = _req()
    summaries = _summaries_with_dup_sources()
    validation = Validation(feasible=False, risks=["No evidence for baselines"], missing=["baselines"])
    artifact = build_artifact(req, summaries, validation)
    # artifact must still be emitted (emit-with-risks) and reflect risks
    assert artifact.validation.feasible is False
    assert len(artifact.demo_spec.acceptance_criteria) >= 2


def test_artifact_created_at_recent():
    """created_at should be recent timestamp. §4.5"""
    import time

    from app.rd.artifacts import build_artifact

    before = time.time()
    artifact = build_artifact(_req(), _summaries_with_dup_sources(), Validation(feasible=True))
    after = time.time()
    assert before <= artifact.created_at <= after


def test_dedupe_empty():
    """dedupe_sources on empty summaries should return empty list."""
    from app.rd.artifacts import dedupe_sources

    assert dedupe_sources([]) == []


def test_demo_spec_model_defaults():
    """DemoSpec defaults must be spec-compliant. §4.5"""
    spec = DemoSpec()
    assert spec.entrypoint == "demo.py"
    assert "python" in spec.tech_stack
