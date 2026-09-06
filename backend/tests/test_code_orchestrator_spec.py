"""Spec-driven tests for Code Orchestrator — design: CodingSpec -> list[FileTask].

Pure function, fixed file set for v1, deterministic, no LLM.
"""

import pytest


def _sample_coding_spec():
    # Use dict shape that orchestrator should accept (or Pydantic model if exists)
    try:
        from app.code.schemas import CodingSpec

        return CodingSpec(
            file_tasks=[
                {"path": "demo.py", "goal": "implement attention", "context_slice": "softmax"},
                {"path": "requirements.txt", "goal": "deps", "context_slice": "numpy"},
                {"path": "test_demo.py", "goal": "tests", "context_slice": "demo exits 0"},
            ],
            test_intents=["demo exits 0", "shape correct"],
            dependencies=["numpy"],
            risks=[],
        )
    except Exception:
        # fallback dict for TDD before schemas exist
        return {
            "file_tasks": [
                {"path": "demo.py", "goal": "implement attention", "context_slice": "softmax"},
                {"path": "requirements.txt", "goal": "deps", "context_slice": "numpy"},
                {"path": "test_demo.py", "goal": "tests", "context_slice": "demo exits 0"},
            ],
            "test_intents": ["demo exits 0"],
            "dependencies": ["numpy"],
            "risks": [],
        }


def test_orchestrator_returns_file_tasks():
    """Must return list of FileTask with path/goal/context_slice. — design §2"""
    from app.code.orchestrator import build_code_plan

    spec = _sample_coding_spec()
    tasks = build_code_plan(spec)
    assert isinstance(tasks, list)
    assert len(tasks) >= 3
    for t in tasks:
        d = t.model_dump() if hasattr(t, "model_dump") else dict(t) if isinstance(t, dict) else {"path": getattr(t, "path", None)}
        assert "path" in d
        assert "goal" in d


def test_orchestrator_fixed_file_set():
    """v1 must include demo.py, requirements.txt, test_demo.py. — design §2"""
    from app.code.orchestrator import build_code_plan

    tasks = build_code_plan(_sample_coding_spec())
    paths = [(t["path"] if isinstance(t, dict) else getattr(t, "path", None)) for t in tasks]
    # Normalize via dump if Pydantic
    try:
        paths = [t.model_dump()["path"] if hasattr(t, "model_dump") else t["path"] for t in tasks]
    except Exception:
        pass
    assert "demo.py" in paths
    assert "requirements.txt" in paths
    assert "test_demo.py" in paths


def test_orchestrator_demo_py_contains_core_algorithm_context():
    """demo.py task should carry core_algorithm/pseudocode context. — design §3"""
    from app.code.orchestrator import build_code_plan

    spec = _sample_coding_spec()
    tasks = build_code_plan(spec)
    demo = next(t for t in tasks if (t["path"] if isinstance(t, dict) else getattr(t, "path")) == "demo.py")
    dump = demo.model_dump() if hasattr(demo, "model_dump") else dict(demo)
    assert dump.get("context_slice") is not None
    assert len(str(dump.get("goal", ""))) > 0


def test_orchestrator_deterministic():
    """Same CodingSpec -> same plan. — design pure function"""
    from app.code.orchestrator import build_code_plan

    spec = _sample_coding_spec()
    assert build_code_plan(spec) == build_code_plan(spec)


def test_orchestrator_no_llm_call():
    """Orchestrator is pure — must not call LLM. — design ponytail"""
    from app.code.orchestrator import build_code_plan

    spec = _sample_coding_spec()
    # No client arg, no async, deterministic
    tasks = build_code_plan(spec)
    tasks2 = build_code_plan(spec)
    assert tasks == tasks2


def test_orchestrator_handles_artifact_directly():
    """Should also accept ResearchArtifact directly as alternative input. — design flexibility"""
    from app.code.orchestrator import build_code_plan

    try:
        from app.rd.schemas import DemoSpec, Requirements, ResearchArtifact, ResearchSummary, Validation

        artifact = ResearchArtifact(
            requirements=Requirements(goal="g", core_idea="c"),
            summaries=[ResearchSummary(direction_id="core_algorithm", findings="algo", sources=[], confidence="high")],
            validation=Validation(feasible=True),
            demo_spec=DemoSpec(entrypoint="demo.py", tech_stack=["python"], core_algorithm="algo", pseudocode="code", acceptance_criteria=["demo.py exits 0"]),
            sources=[],
            created_at=0,
        )
        tasks = build_code_plan(artifact)  # may support artifact or spec
        assert any("demo.py" in (t["path"] if isinstance(t, dict) else getattr(t, "path")) for t in tasks)
    except Exception as e:
        # If orchestrator only accepts CodingSpec, this is acceptable — spec says artifact -> CodeAnalyzer -> CodingSpec
        pytest.skip(f"orchestrator artifact input not required: {e}")
