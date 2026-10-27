"""Spec-driven tests for Code Extend mode (offline).

Design: idea + repo -> analyzer -> orchestrator -> subagents -> tester -> repo sandbox.
Streaming NDJSON: extend_start, context, plan, change_start|change_result,
tester/result, sandbox/start|result, code/artifact.
"""

import json

import pytest

from app.code.schemas import ExtendFileTask, ExtendSpec, GeneratedFile, RepoContext

CORE = "def add(a, b):\n    return a + b\n"
TEST_CORE = "from pkg.core import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
MODIFIED_CORE = "def add(a, b):\n    return a + b\n\n\ndef square(x):\n    return x * x\n"
CREATE_CONTENT = "def square(x):\n    return x * x\n"
TEST_EXTEND_CONTENT = "from pkg.extra import square\n\n\ndef test_square():\n    assert square(3) == 9\n"


def _mini_repo(root):
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "core.py").write_text(CORE, encoding="utf-8")
    (root / "tests" / "test_core.py").write_text(TEST_CORE, encoding="utf-8")
    return root


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResp:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


def _make_extend_client():
    """Serves analyzer json plan, change-subagent text, and tester text by prompt."""

    plan = {
        "entrypoint": "",
        "focus_paths": ["pkg/extra.py"],
        "file_tasks": [
            {"path": "pkg/extra.py", "action": "create", "goal": "square helper", "content": ""},
            {"path": "pkg/core.py", "action": "modify", "goal": "integrate square", "content": ""},
        ],
        "test_intents": ["square(3) == 9"],
        "dependencies": [],
        "risks": [],
    }

    class _Completions:
        async def create(self, **kwargs):
            if kwargs.get("response_format") == {"type": "json_object"}:
                return _FakeResp(json.dumps(plan))
            prompt = " ".join(str(m.get("content", "")) for m in kwargs.get("messages", [])).lower()
            if "test_extend_poc.py" in prompt or "test_intents" in prompt:
                return _FakeResp(TEST_EXTEND_CONTENT)
            if "modify" in prompt:
                return _FakeResp(MODIFIED_CORE)
            return _FakeResp(CREATE_CONTENT)

    return type("C", (), {"chat": type("C2", (), {"completions": _Completions()})()})()


def _failing_client():
    class _Completions:
        async def create(self, **kwargs):
            raise RuntimeError("LLM down")

    return type("C", (), {"chat": type("C2", (), {"completions": _Completions()})()})()


class _JsonClient:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload)

    def _make(self):
        raw = self._raw

        class _Completions:
            async def create(self, **kwargs):
                return _FakeResp(raw)

        return type("C", (), {"chat": type("C2", (), {"completions": _Completions()})()})()

    def __getattr__(self, item):
        return getattr(self._make(), item)


@pytest.mark.asyncio
async def test_extend_pipeline_event_sequence(monkeypatch, tmp_path):
    """Full extend stream: start -> context -> plan -> changes -> tester -> sandbox -> artifact."""
    from unittest.mock import AsyncMock

    from app.code.extend_pipeline import code_extend_stream
    from app.code.schemas import CodeExtendRequest

    monkeypatch.setattr("app.core.db.code_run_create", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.code_run_update", AsyncMock(), raising=False)
    monkeypatch.setattr("app.core.db.code_artifact_upsert", AsyncMock(), raising=False)

    repo = _mini_repo(tmp_path)
    client = _make_extend_client()

    events = [e async for e in code_extend_stream(client, CodeExtendRequest(idea="add square helper", path=str(repo)))]

    assert events[0]["event"] == "extend_start"
    assert any(e["event"] == "context" for e in events)
    plan_events = [e for e in events if e["event"] == "plan"]
    assert plan_events and len(plan_events[0]["file_tasks"]) == 2
    assert len([e for e in events if e["event"] == "change_start"]) >= 2
    assert len([e for e in events if e["event"] == "change_result"]) >= 2
    assert any(e["type"] == "tester" and e["event"] == "result" for e in events)
    assert any(e["type"] == "sandbox" and e["event"] == "start" for e in events)
    assert any(e["type"] == "sandbox" and e["event"] == "result" for e in events)

    last = events[-1]
    assert last["type"] == "code" and last["event"] == "artifact"
    data = last["data"]
    assert len(data["changes"]) == 2
    assert all(c["content"] for c in data["changes"])
    assert data["sandbox"]["passed"] is True


@pytest.mark.asyncio
async def test_extend_pipeline_error_no_repo():
    """No repo_id/path -> extend_error mentioning repo_id or path, no artifact."""
    from app.code.extend_pipeline import code_extend_stream

    events = [e async for e in code_extend_stream(_make_extend_client(), {"idea": "x"})]

    errors = [e for e in events if e["event"] == "extend_error"]
    assert errors and "repo_id or path" in errors[0]["message"]
    assert not any(e["event"] == "artifact" for e in events)


@pytest.mark.asyncio
async def test_extend_analyzer_fallback():
    """LLM failure -> deterministic create extend_poc.py fallback with a risk note."""
    from app.code.extend_analyzer import analyze_extension

    spec = await analyze_extension(_failing_client(), "add caching", RepoContext())

    assert spec.file_tasks[0].path == "extend_poc.py"
    assert spec.file_tasks[0].action == "create"
    assert "fallback" in spec.risks[0].lower()


@pytest.mark.asyncio
async def test_extend_analyzer_grounded_modify():
    """Planner modify task keeps its provided content; entrypoint of a changed path is preserved."""
    from app.code.extend_analyzer import analyze_extension

    client = _JsonClient({
        "entrypoint": "pkg/core.py",
        "file_tasks": [
            {"path": "pkg/core.py", "action": "modify", "goal": "g", "content": "def add(a, b):\n    return a + b  # extended\n"}
        ],
        "test_intents": ["t"],
        "dependencies": [],
        "risks": [],
    })
    spec = await analyze_extension(client, "extend add", RepoContext())

    assert spec.file_tasks[0].action == "modify"
    assert spec.file_tasks[0].content
    assert spec.entrypoint == "pkg/core.py"


def test_extend_orchestrator_plan_ordering():
    """Plan is deduped, unsafe dropped, modifies first, goals defaulted, capped."""
    from app.code.extend_orchestrator import build_extend_plan

    spec = ExtendSpec(
        file_tasks=[
            ExtendFileTask(path="../evil.py", action="modify", goal="evil"),
            ExtendFileTask(path="a.py", action="create", goal=""),
            ExtendFileTask(path="a.py", action="create", goal="dup"),
            ExtendFileTask(path="m.py", action="modify", goal="m"),
            ExtendFileTask(path="z.py", action="create", goal=""),
        ],
    )
    plan = build_extend_plan(spec, RepoContext())
    paths = [t.path for t in plan]

    assert "../evil.py" not in paths
    assert paths.count("a.py") == 1
    assert paths.index("m.py") < paths.index("z.py")
    assert all(t.goal for t in plan)
    assert len(build_extend_plan(spec, RepoContext(), max_changes=2)) == 2


@pytest.mark.asyncio
async def test_extend_subagent_fills_modify_content(tmp_path):
    """Modify task gets the current file content in its prompt and returns updated content."""
    from app.code.extend_subagent import run_change_task

    captured = {}

    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResp("def f():\n    return 2  # changed\n")

    client = type("C", (), {"chat": type("C2", (), {"completions": _Completions()})()})()
    ctx = RepoContext(files={"mod.py": "def f():\n    return 1\n"})
    task = ExtendFileTask(path="mod.py", action="modify", goal="change f", content="")

    result = await run_change_task(client, task, ctx)

    assert "return 2" in result.content
    prompt = " ".join(str(m.get("content", "")) for m in captured.get("messages", []))
    assert "return 1" in prompt


@pytest.mark.asyncio
async def test_extend_tester_fallback():
    """Tester LLM failure -> fallback file asserting the PoC entrypoint exists."""
    from app.code.extend_tester import generate_extend_tests

    result = await generate_extend_tests(_failing_client(), "idea", ExtendSpec(entrypoint="extend_poc.py"), RepoContext())

    assert isinstance(result, GeneratedFile)
    assert result.path == "test_extend_poc.py"
    assert "test_poc_module_present" in result.content
    assert "extend_poc.py" in result.content


@pytest.mark.asyncio
async def test_repo_sandbox_mini_integration(tmp_path):
    """Sandbox applies changes on a copy and leaves the real repo pristine."""
    from app.code.repo_sandbox import run_repo_sandbox
    from app.code.schemas import ExtendFileTask

    repo = _mini_repo(tmp_path)
    (repo / ".git").mkdir()
    original_core = (repo / "pkg" / "core.py").read_text(encoding="utf-8")
    ctx_files = {"pkg/core.py": CORE, "tests/test_core.py": TEST_CORE}

    tasks = [
        ExtendFileTask(
            path="pkg/core.py",
            action="modify",
            search_replace=[{"search": "def add(a, b):", "replace": "def add(a, b, c=0):"}],
        ),
        ExtendFileTask(path="pkg/extra.py", action="create", content=CREATE_CONTENT),
        ExtendFileTask(path="../evil.py", action="create", content="evil"),
    ]

    result, changes = await run_repo_sandbox(repo, ctx_files, tasks, timeout=60)

    assert result.passed is True
    assert len(changes) == 3
    assert sum(1 for c in changes if c["applied"]) == 2
    assert sum(1 for c in changes if not c["applied"]) == 1
    # Real repo untouched
    assert (repo / "pkg" / "core.py").read_text(encoding="utf-8") == original_core
    assert not (repo / "pkg" / "extra.py").exists()


@pytest.mark.asyncio
async def test_repo_sandbox_failing_tests(tmp_path):
    """Failing repo tests -> passed False with failure in the pytest log."""
    from app.code.repo_sandbox import run_repo_sandbox

    repo = _mini_repo(tmp_path)
    (repo / "tests" / "test_core.py").write_text(
        "from pkg.core import add\n\n\ndef test_add():\n    assert add(2, 3) == 999\n",
        encoding="utf-8",
    )

    result, _changes = await run_repo_sandbox(repo, None, [], timeout=60)

    assert result.passed is False
    assert "failed" in result.pytest_log or "1 failed" in result.pytest_log


def test_patcher_guards(tmp_path):
    """Empty modify is a no-op, missed search blocks fail, paths are normalized safely."""
    from app.code.patcher import apply_changes
    from app.code.schemas import ExtendFileTask

    (tmp_path / "f.py").write_text("original", encoding="utf-8")
    (tmp_path / "g.py").write_text("abc", encoding="utf-8")

    tasks = [
        ExtendFileTask(path="f.py", action="modify", content="", search_replace=[]),
        ExtendFileTask(path="g.py", action="modify", content="", search_replace=[{"search": "zzz", "replace": "y"}]),
        ExtendFileTask(path="a/b/c.py", action="create", content="x"),
        ExtendFileTask(path="./x.py", action="create", content="y"),
        ExtendFileTask(path="/abs.py", action="create", content="z"),
    ]
    changes, _log = apply_changes(tmp_path, tasks)
    by_path = {c["path"]: c for c in changes}

    assert by_path["f.py"]["applied"] is False
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "original"
    assert by_path["g.py"]["applied"] is False
    assert (tmp_path / "g.py").read_text(encoding="utf-8") == "abc"
    # nested create works
    assert (tmp_path / "a" / "b" / "c.py").exists()
    # "./x.py" lands inside root
    assert by_path["./x.py"]["applied"] is True
    assert (tmp_path / "x.py").exists()
    # "/abs.py" is stripped to a repo-relative path, not the real filesystem root
    assert by_path["abs.py"]["applied"] is True
    assert (tmp_path / "abs.py").exists()


@pytest.mark.asyncio
async def test_repo_context_local_mode(tmp_path):
    """A local path builds a mode=local context with lexical hits ranked."""
    from app.code.repo_context import build_repo_context

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text("def square(x):\n    return x * x\n", encoding="utf-8")

    ctx = await build_repo_context("add a square helper to pkg", path=str(tmp_path))

    assert ctx.mode == "local"
    assert ctx.files
    assert ctx.hits and ctx.hits[0]["path"] == "pkg/core.py"


@pytest.mark.asyncio
async def test_repo_context_none_mode():
    """No repo_id and no path -> mode none."""
    from app.code.repo_context import build_repo_context

    ctx = await build_repo_context("idea", repo_id=None, path=None)

    assert ctx.mode == "none"
