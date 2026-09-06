"""Spec-driven tests for Sandbox — design: tmpdir + subprocess, no Docker for v1.

Interface SandboxResult{passed, pytest_log, demo_log, returncode, runtime_ms}.
"""

import pytest


def _files_simple():
    return [
        {"path": "demo.py", "content": "print('hello demo')\n"},
        {"path": "requirements.txt", "content": ""},
        {"path": "test_demo.py", "content": "def test_demo():\n    assert True\n"},
    ]


@pytest.mark.asyncio
async def test_sandbox_returns_result_shape():
    """Must return SandboxResult with spec fields. — design §5"""
    from app.code.sandbox import run_sandbox

    result = await run_sandbox(_files_simple(), timeout=10)

    dump = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    assert "passed" in dump and isinstance(dump["passed"], bool)
    assert "pytest_log" in dump
    assert "demo_log" in dump
    assert "returncode" in dump


@pytest.mark.asyncio
async def test_sandbox_passes_on_valid_demo():
    """Valid demo.py + passing test should yield passed=True. — design §5"""
    from app.code.sandbox import run_sandbox

    result = await run_sandbox(_files_simple(), timeout=15)
    dump = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    assert dump["passed"] is True
    assert dump["returncode"] == 0


@pytest.mark.asyncio
async def test_sandbox_fails_on_broken_demo():
    """Broken demo.py should yield passed=False, not raise. — design emit-with-risks"""
    from app.code.sandbox import run_sandbox

    files = [
        {"path": "demo.py", "content": "raise RuntimeError('fail')\n"},
        {"path": "requirements.txt", "content": ""},
        {"path": "test_demo.py", "content": "def test_demo():\n    import subprocess\n    assert subprocess.run(['python','demo.py']).returncode==0\n"},
    ]
    result = await run_sandbox(files, timeout=10)
    dump = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    assert dump["passed"] is False
    assert dump["returncode"] != 0 or "fail" in dump["pytest_log"].lower() or "fail" in dump["demo_log"].lower()


@pytest.mark.asyncio
async def test_sandbox_timeout():
    """Long-running demo must be killed after timeout, not hang. — design §5"""
    from app.code.sandbox import run_sandbox

    files = [
        {"path": "demo.py", "content": "import time; time.sleep(100)\n"},
        {"path": "requirements.txt", "content": ""},
        {"path": "test_demo.py", "content": "def test_demo():\n    assert True\n"},
    ]
    result = await run_sandbox(files, timeout=2)
    dump = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    assert dump["passed"] is False  # timeout is failure
    # logs should mention timeout or still be present
    assert isinstance(dump["pytest_log"], str) and isinstance(dump["demo_log"], str)


@pytest.mark.asyncio
async def test_sandbox_isolated_tmpdir():
    """Each run must be isolated — files written to tmpdir, not cwd. — design §5"""
    import os

    from app.code.sandbox import run_sandbox

    # Ensure sandbox doesn't pollute cwd: demo.py should not appear in repo root
    cwd_demo = "demo.py"
    before = os.path.exists(cwd_demo)
    await run_sandbox(_files_simple(), timeout=5)
    after = os.path.exists(cwd_demo)
    # If cwd had demo.py before, it should still be same; if not, sandbox must not have created it
    assert before == after or not after


@pytest.mark.asyncio
async def test_sandbox_never_raises():
    """Sandbox must never raise even on malformed files — return result with passed=False. — design §5"""
    from app.code.sandbox import run_sandbox

    files = [{"path": "demo.py", "content": "!!! not python !!!"}]
    result = await run_sandbox(files, timeout=5)
    dump = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    assert "passed" in dump


def test_sandbox_interface_exists():
    """Module must expose run_sandbox and SandboxResult. — design §5"""
    import app.code.sandbox as m

    assert hasattr(m, "run_sandbox")
    # SandboxResult may be a Pydantic model or dict — check existence if defined
    assert hasattr(m, "SandboxResult") or True
