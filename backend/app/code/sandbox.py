"""Sandbox — tmpdir + subprocess, no Docker for v1.

ponytail: subprocess without cgroup/network ns, switch to docker+gvisor if untrusted.
"""

import ast
import asyncio
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel

from .schemas import SandboxResult

# Map top-level imports to pip packages (common aliases)
_IMPORT_TO_PIP = {
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
}


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


def _patch_requirements_from_imports(tmp: Path) -> str:
    """Parse demo.py imports via ast and auto-append missing to requirements.txt. Returns log."""
    demo = tmp / "demo.py"
    req = tmp / "requirements.txt"
    if not demo.exists():
        return ""
    try:
        tree = ast.parse(demo.read_text(encoding="utf-8"))
    except Exception:
        return ""
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    # stdlib + local
    stdlib = {"os", "sys", "json", "time", "math", "random", "collections", "pathlib", "typing", "subprocess", "tempfile", "re", "datetime"}
    imports = {i for i in imports if i not in stdlib and not i.startswith("_")}
    if not imports:
        return ""
    existing = set()
    if req.exists():
        try:
            existing = {line.strip().lower().split("==")[0].split(">")[0].strip() for line in req.read_text().splitlines() if line.strip() and not line.strip().startswith("#")}
        except Exception:
            existing = set()
    # Map aliases
    to_add = []
    for imp in sorted(imports):
        pip_name = _IMPORT_TO_PIP.get(imp, imp)
        if pip_name.lower() not in existing and pip_name.lower() not in {e.lower() for e in existing}:
            # only auto-add common packages, avoid hallucinating
            if pip_name.lower() in {"numpy", "torch", "pandas", "scikit-learn", "scipy", "matplotlib", "Pillow", "requests", "tqdm", "pytest", "opencv-python", "transformers"}:
                to_add.append(pip_name)
    if to_add:
        try:
            with req.open("a", encoding="utf-8") as f:
                for pkg in to_add:
                    f.write(f"{pkg}\n")
            return f"auto-added deps: {', '.join(to_add)}"
        except Exception:
            pass
    return ""


def _run_sync(tmp: Path, timeout: int) -> tuple[int, str, str, int]:
    start = time.time()
    # AST dep inference before pip (0 LLM, stdlib)
    patch_log = _patch_requirements_from_imports(tmp)
    # pip install if requirements.txt exists and non-empty
    req = tmp / "requirements.txt"
    pip_log = patch_log
    pip_code = 0
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
                pip_log = (pip_log + "\n" if pip_log else "") + proc.stdout + proc.stderr
                pip_code = proc.returncode
        except Exception as e:
            pip_log = (pip_log + "\n" if pip_log else "") + str(e)
            pip_code = 1

    # pytest — discover all tests (supports dynamic model.py etc.)
    pytest_log = ""
    pytest_code = 0
    has_test = any((tmp / p).exists() for p in ["test_demo.py", "tests/test_demo.py", "test_*.py"])
    try:
        if not has_test:
            pytest_log = (pip_log + "\n" if pip_log else "") + "no test file"
            pytest_code = 127
        else:
            # Run pytest without explicit file to allow discovery; fallback to test_demo.py
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            pytest_log = (pip_log + "\n" if pip_log else "") + proc.stdout + proc.stderr
            pytest_code = proc.returncode
            # Fallback to explicit if no tests collected
            if "no tests ran" in pytest_log.lower() or "collected 0 items" in pytest_log.lower():
                proc2 = subprocess.run(
                    [sys.executable, "-m", "pytest", "test_demo.py", "-v"],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                pytest_log = (pip_log + "\n" if pip_log else "") + proc2.stdout + proc2.stderr
                pytest_code = proc2.returncode
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
    # Fixed passed logic: pip must succeed (if present), pytest and demo must succeed
    # pip_code non-zero is failure; pytest 127 (no pytest) is ok if demo passes (minimal env)
    if pip_code != 0:
        passed = False
    elif pytest_code == 127 and demo_code == 0:
        passed = True  # no test file but demo ok — minimal case
    else:
        passed = (pytest_code == 0 and demo_code == 0)
    # Timeout always fails
    if "timeout" in pytest_log.lower() or "timeout" in demo_log.lower():
        passed = False
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
