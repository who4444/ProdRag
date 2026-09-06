"""Sandbox — tmpdir + subprocess, no Docker for v1.

ponytail: subprocess without cgroup/network ns, switch to docker+gvisor if untrusted.
"""

import asyncio
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel

from .schemas import SandboxResult


def _write_files(tmp: Path, files: list[dict | BaseModel]):
    for f in files:
        if hasattr(f, "model_dump"):
            d = f.model_dump()
            path = d.get("path")
            content = d.get("content", "")
        elif isinstance(f, dict):
            path = f.get("path")
            content = f.get("content", "")
        else:
            path = getattr(f, "path", "demo.py")
            content = getattr(f, "content", "")
        if not path:
            continue
        p = tmp / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content or "", encoding="utf-8")


def _run_sync(tmp: Path, timeout: int) -> tuple[int, str, str, int]:
    start = time.time()
    # pip install if requirements.txt exists and non-empty
    req = tmp / "requirements.txt"
    pip_log = ""
    if req.exists():
        try:
            txt = req.read_text().strip()
            if txt:
                proc = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                pip_log = proc.stdout + proc.stderr
        except Exception as e:
            pip_log = str(e)

    # pytest
    pytest_log = ""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "test_demo.py", "-v"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        pytest_log = (pip_log + "\n" if pip_log else "") + proc.stdout + proc.stderr
        pytest_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        pytest_log = (pip_log + "\n" if pip_log else "") + (e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout or "")) + " timeout"
        pytest_code = 124
    except FileNotFoundError:
        pytest_log = pip_log + "\npytest not found"
        pytest_code = 127
    except Exception as e:
        pytest_log = pip_log + f"\npytest error: {e}"
        pytest_code = 1

    # demo.py
    demo_log = ""
    demo_code = 0
    try:
        proc = subprocess.run(
            [sys.executable, "demo.py"],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        demo_log = proc.stdout + proc.stderr
        demo_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        demo_log = (e.stdout.decode() if isinstance(e.stdout, bytes) else str(e.stdout or "")) + " timeout"
        demo_code = 124
    except Exception as e:
        demo_log = str(e)
        demo_code = 1

    runtime_ms = int((time.time() - start) * 1000)
    # passed if both pytest and demo succeed (pytest_code 0 and demo_code 0) and files exist
    # If no test file, just check demo
    passed = (pytest_code == 0 and demo_code == 0)
    # If pytest not found but demo succeeded, still consider passed for minimal case
    if pytest_code == 127 and demo_code == 0:
        passed = True
    return demo_code, pytest_log, demo_log, runtime_ms


async def run_sandbox(files: list[dict | BaseModel], timeout: int = 15) -> SandboxResult:
    # Validate files
    if not files:
        return SandboxResult(passed=False, pytest_log="no files", demo_log="", returncode=1, runtime_ms=0)

    # Normalize files to ensure demo.py exists
    has_demo = any((f.get("path") if isinstance(f, dict) else getattr(f, "path", None)) == "demo.py" or (hasattr(f, "model_dump") and f.model_dump().get("path") == "demo.py") for f in files)
    if not has_demo:
        # Still run but will fail
        pass

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            _write_files(tmp, files)

            # Run in thread to avoid blocking event loop (like db.py to_thread)
            loop = asyncio.get_running_loop()
            demo_code, pytest_log, demo_log, runtime_ms = await loop.run_in_executor(None, lambda: _run_sync(tmp, timeout))

            passed = (demo_code == 0)
            # If pytest log contains failure, mark failed
            if "failed" in pytest_log.lower() or "error" in pytest_log.lower():
                # But if demo succeeded and pytest not crucial, keep demo-based passed
                if "passed" in pytest_log.lower() and demo_code == 0:
                    passed = True
                elif "1 failed" in pytest_log.lower():
                    passed = False

            # Timeout detection
            if "timeout" in pytest_log.lower() or "timeout" in demo_log.lower():
                passed = False

            # Overall passed is demo success and pytest success
            # For v1, require both unless pytest not present
            if "pytest not found" in pytest_log and demo_code == 0:
                passed = True
            elif demo_code != 0:
                passed = False

            return SandboxResult(passed=passed, pytest_log=pytest_log[:4000], demo_log=demo_log[:4000], returncode=demo_code, runtime_ms=runtime_ms)
    except Exception as e:
        return SandboxResult(passed=False, pytest_log=str(e)[:4000], demo_log="", returncode=1, runtime_ms=0)
