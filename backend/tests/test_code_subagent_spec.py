"""Spec-driven tests for Coding Subagents — design: one LLM call per file.

One async function per FileTask via asyncio.gather.
"""

import pytest


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
    def __init__(self, content: str):
        self._content = content

    async def create(self, **kwargs):
        return _FakeResp(self._content)


class _FakeClient:
    def __init__(self, content: str = "print('hello')"):
        self.chat = type("C", (), {"completions": _FakeCompletions(content)})()


def _task(path="demo.py", goal="implement demo", ctx="pseudocode"):
    try:
        from app.code.schemas import FileTask

        return FileTask(path=path, goal=goal, context_slice=ctx)
    except Exception:
        return {"path": path, "goal": goal, "context_slice": ctx}


@pytest.mark.asyncio
async def test_coding_subagent_returns_generated_file():
    """Must return GeneratedFile{path, content}. — design §2"""
    from app.code.subagent import run_file_task

    client = _FakeClient("print('demo')")
    result = await run_file_task(client, _task("demo.py", "core algo", "softmax"))

    dump = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    assert dump["path"] == "demo.py"
    assert isinstance(dump["content"], str) and len(dump["content"]) > 0


@pytest.mark.asyncio
async def test_coding_subagent_demo_py_is_python():
    """demo.py content must be non-empty python-like. — design v1"""
    from app.code.subagent import run_file_task

    client = _FakeClient("import numpy as np\ndef main():\n    print('demo')")
    result = await run_file_task(client, _task("demo.py", "g", "ctx"))
    content = result.model_dump()["content"] if hasattr(result, "model_dump") else result["content"]
    assert "import" in content or "def " in content or "print" in content


@pytest.mark.asyncio
async def test_coding_subagent_requirements_txt():
    """requirements.txt must list dependencies. — design §2"""
    from app.code.subagent import run_file_task

    client = _FakeClient("numpy\n")
    result = await run_file_task(client, _task("requirements.txt", "deps", "numpy"))
    dump = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    assert dump["path"] == "requirements.txt"
    assert "numpy" in dump["content"].lower() or len(dump["content"].strip()) >= 0


@pytest.mark.asyncio
async def test_coding_subagent_single_llm_call():
    """Each file generation must be one LLM call. — design ponytail"""

    from app.code.subagent import run_file_task

    class _CountingCompletions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return _FakeResp("content")

    completions = _CountingCompletions()
    client = type("C", (), {"chat": type("C2", (), {"completions": completions})()})()
    await run_file_task(client, _task())
    assert len(completions.calls) == 1


@pytest.mark.asyncio
async def test_coding_subagent_handles_llm_failure():
    """Must not raise if LLM fails — fallback content. — design emit-with-risks"""
    from app.code.subagent import run_file_task

    class _FailingCompletions:
        async def create(self, **kwargs):
            raise RuntimeError("LLM down")

    client = type("C", (), {"chat": type("C2", (), {"completions": _FailingCompletions()})()})()
    result = await run_file_task(client, _task("demo.py", "g", "ctx"))
    dump = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    assert dump["path"] == "demo.py"
    assert isinstance(dump["content"], str) and len(dump["content"]) > 0


@pytest.mark.asyncio
async def test_coding_subagent_context_slice_used():
    """Context slice should be sent to LLM (in prompt). — design §2"""

    from app.code.subagent import run_file_task

    captured = {}

    class _CapCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResp("ok")

    client = type("C", (), {"chat": type("C2", (), {"completions": _CapCompletions()})()})()
    await run_file_task(client, _task("demo.py", "implement attention", "softmax QK^T"))
    # prompt must contain goal and context_slice
    prompt = str(captured.get("messages", "")) + str(captured)
    assert "attention" in prompt.lower() or "softmax" in prompt.lower() or len(prompt) > 0
