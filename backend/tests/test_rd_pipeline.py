from app.rd.orchestrator import TAXONOMY, build_plan
from app.rd.artifacts import build_artifact, dedupe_sources
from app.rd.schemas import Requirements, ResearchSummary, Validation
from app.rd.validator import _fallback_validation  # type: ignore


def test_orchestrator_fixed_taxonomy():
    req = Requirements(goal="demo attention", core_idea="attention is all you need", demo_type="script")
    dirs = build_plan(req)
    assert len(dirs) == 5
    assert [d.id for d in dirs] == [t["id"] for t in TAXONOMY]
    assert all(d.question for d in dirs)


def test_validator_fallback():
    req = Requirements(goal="x", core_idea="y")
    summaries = [
        ResearchSummary(direction_id="core_algorithm", findings="found", sources=[], confidence="high"),
        ResearchSummary(direction_id="datasets", findings="", sources=[], confidence="low"),
    ]
    v = _fallback_validation(req, summaries)
    assert v.coverage["core_algorithm"] is True
    assert v.coverage["datasets"] is False
    assert "datasets" in v.missing


def test_dedupe_and_artifact():
    s1 = ResearchSummary(direction_id="core_algorithm", findings="algo", sources=[{"source": "a", "page": 1, "content": "hello"}], confidence="high")
    s2 = ResearchSummary(direction_id="core_algorithm", findings="algo2", sources=[{"source": "a", "page": 1, "content": "hello"}], confidence="high")
    assert len(dedupe_sources([s1, s2])) == 1
    req = Requirements(goal="g", core_idea="c")
    v = Validation(coverage={}, feasible=True)
    art = build_artifact(req, [s1], v)
    assert art.demo_spec.entrypoint == "demo.py"
    assert art.sources
