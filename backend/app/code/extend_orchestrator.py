"""Extend Orchestrator — ExtendSpec/idea+ctx -> list[ExtendFileTask].

Pure function, no LLM, deterministic; fills gaps left by the analyzer.
"""

from .schemas import ExtendFileTask, RepoContext


def build_extend_plan(spec, ctx: RepoContext | None = None, max_changes: int = 3) -> list[ExtendFileTask]:
    """Normalize + order the plan: modifications first, creates last, unsafe paths dropped."""
    if hasattr(spec, "file_tasks"):
        tasks = [ExtendFileTask(**t.model_dump()) for t in spec.file_tasks]
    elif isinstance(spec, dict) and "file_tasks" in spec:
        tasks = [ExtendFileTask(**t) if isinstance(t, dict) else t for t in spec.get("file_tasks", [])]
    else:
        idea = str(spec if not isinstance(spec, dict) else spec.get("idea", "extension"))
        tasks = [ExtendFileTask(path="extend_poc.py", action="create", goal=idea[:200], content="")]
    safe: list[ExtendFileTask] = []
    seen: set[str] = set()
    for t in tasks:
        rel = t.path.strip().lstrip("/")
        if not rel or rel.startswith("..") or "../" in rel:
            continue
        if rel in seen:
            continue
        seen.add(rel)
        if not t.goal:
            t.goal = f"{t.action} {rel}"
        safe.append(t)
    safe.sort(key=lambda t: (0 if t.action == "modify" else 1, t.path))
    return safe[:max_changes]
