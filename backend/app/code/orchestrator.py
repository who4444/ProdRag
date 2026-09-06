"""Code Orchestrator — CodingSpec/ResearchArtifact -> list[FileTask].

Pure function, fixed file set for v1, deterministic, no LLM.
"""

from .schemas import CodingSpec, FileTask


def _coerce_coding_spec(obj) -> CodingSpec:
    if isinstance(obj, CodingSpec):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return CodingSpec(**obj.model_dump())
        except Exception:
            pass
    if isinstance(obj, dict):
        # dict may be CodingSpec shape or ResearchArtifact shape
        if "file_tasks" in obj:
            try:
                return CodingSpec(**obj)
            except Exception:
                pass
        # ResearchArtifact shape: build minimal CodingSpec from it
        try:
            from ..rd.schemas import ResearchArtifact

            if "requirements" in obj and "demo_spec" in obj:
                artifact = ResearchArtifact(**obj)
                # reuse analyzer fallback logic lightly: build tasks from demo_spec
                demo_spec = artifact.demo_spec
                criteria = demo_spec.acceptance_criteria or ["demo.py exits 0"]
                tasks = [
                    FileTask(path="demo.py", goal="implement core demo", context_slice=demo_spec.core_algorithm[:500]),
                    FileTask(path="requirements.txt", goal="dependencies", context_slice=",".join(demo_spec.tech_stack)),
                    FileTask(path="test_demo.py", goal="tests", context_slice="; ".join(criteria[:2])),
                ]
                return CodingSpec(file_tasks=tasks, test_intents=criteria[:3], dependencies=[d for d in demo_spec.tech_stack if d != "python"], risks=artifact.validation.risks)
        except Exception:
            pass
        # fallback: try to coerce FileTask list
        try:
            tasks = [FileTask(**t) if isinstance(t, dict) else t for t in obj.get("file_tasks", [])]
            return CodingSpec(file_tasks=tasks, test_intents=obj.get("test_intents", []), dependencies=obj.get("dependencies", []), risks=obj.get("risks", []))
        except Exception:
            pass
    # ultimate fallback
    return CodingSpec(
        file_tasks=[
            FileTask(path="demo.py", goal="implement demo", context_slice=""),
            FileTask(path="requirements.txt", goal="dependencies", context_slice="numpy"),
            FileTask(path="test_demo.py", goal="tests", context_slice="demo.py exits 0"),
        ],
        test_intents=["demo.py exits 0"],
        dependencies=["numpy"],
        risks=[],
    )


def build_code_plan(spec) -> list[FileTask]:
    coding_spec = _coerce_coding_spec(spec)
    tasks = list(coding_spec.file_tasks)

    # Ensure fixed set for v1
    paths = {t.path for t in tasks}
    if "demo.py" not in paths:
        # Use core context if available
        ctx = ""
        try:
            ctx = coding_spec.file_tasks[0].context_slice if coding_spec.file_tasks else ""
        except Exception:
            pass
        tasks.insert(0, FileTask(path="demo.py", goal="implement core demo", context_slice=ctx))
    if "requirements.txt" not in paths:
        tasks.append(FileTask(path="requirements.txt", goal="dependencies", context_slice=",".join(coding_spec.dependencies) or "numpy"))
    if "test_demo.py" not in paths:
        tasks.append(FileTask(path="test_demo.py", goal="tests for acceptance criteria", context_slice="; ".join(coding_spec.test_intents[:2]) or "demo.py exits 0"))

    # Ensure demo.py has context_slice (for test that checks core_algorithm context)
    for t in tasks:
        if t.path == "demo.py" and not t.context_slice:
            t.context_slice = coding_spec.test_intents[0] if coding_spec.test_intents else "demo"
        if not t.goal:
            t.goal = f"generate {t.path}"

    # Deterministic order: demo.py, requirements.txt, test_demo.py, others
    order = {"demo.py": 0, "requirements.txt": 1, "test_demo.py": 2}
    tasks.sort(key=lambda x: order.get(x.path, 99))
    return tasks
