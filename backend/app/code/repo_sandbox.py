"""Extend-mode repo sandbox — copy the target repo subset to a temp dir, apply
the planned changes there, then run pytest + optional entrypoint.

Never modifies the real repo: the PoC validates changes on a pristine copy.
Never raises — modeled on code/sandbox.py.
"""

import asyncio
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .patcher import apply_changes
from .schemas import ExtendFileTask, SandboxResult

logger = logging.getLogger(__name__)

_IGNORE_DIRS = [
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".nox", "dist", "build", ".idea", ".vscode",
    ".egg-info",
]


def _copy_subset(root: Path, files: dict[str, str] | None, dst: Path) -> list[str]:
    """Copy only the context files (or the whole tree when files=None) into dst.

    Returns list of copied repo-relative paths.
    """
    copied: list[str] = []
    if files is None:
        shutil.copytree(
            root,
            dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*_IGNORE_DIRS),
        )
        for p in dst.rglob("*"):
            if p.is_file():
                copied.append(p.relative_to(dst).as_posix())
        return copied
    for rel, text in files.items():
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        copied.append(rel)
    return copied


def _run_sync(dst: Path, entrypoint: str, pip_timeout: int, pytest_timeout: int) -> tuple[int, int, str, str, int]:
    """Run pytest + optional entrypoint in the repo copy.

    Returns (entry_code, pytest_code, pytest_log, entry_log, runtime_ms).
    """
    start = time.time()
    pytest_log = ""
    entry_log = ""
    pytest_code = 0
    entry_code = 0

    req = dst / "requirements.txt"
    if req.exists():
        try:
            if req.read_text(encoding="utf-8").strip():
                proc = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
                    cwd=dst,
                    capture_output=True,
                    text=True,
                    timeout=pip_timeout,
                )
                pytest_log = proc.stdout + proc.stderr
                if proc.returncode != 0:
                    pytest_log += "\npip failed"
        except Exception as e:
            pytest_log += f"\npip error: {e}"

    has_tests = any(dst.rglob("test_*.py")) or any(dst.rglob("*_test.py"))
    if has_tests:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=dst,
                capture_output=True,
                text=True,
                timeout=pytest_timeout,
            )
            pytest_log = (pytest_log + "\n" if pytest_log else "") + proc.stdout + proc.stderr
            pytest_code = proc.returncode
        except subprocess.TimeoutExpired:
            pytest_log += "\npytest timeout"
            pytest_code = 124
        except FileNotFoundError:
            pytest_log += "\npytest not found"
            pytest_code = 127
    else:
        pytest_log += "\nno test_*.py files in repo copy"
        pytest_code = 127

    ep = (entrypoint or "").strip().lstrip("/")
    if ep:
        target = dst / ep
        try:
            if not target.exists():
                entry_log = f"entrypoint missing: {ep}"
                entry_code = 127
            else:
                proc = subprocess.run(
                    [sys.executable, ep],
                    cwd=dst,
                    capture_output=True,
                    text=True,
                    timeout=pytest_timeout,
                )
                entry_log = proc.stdout + proc.stderr
                entry_code = proc.returncode
        except subprocess.TimeoutExpired:
            entry_log = "entrypoint timeout"
            entry_code = 124
        except Exception as e:
            entry_log = f"entrypoint error: {e}"
            entry_code = 1

    runtime_ms = int((time.time() - start) * 1000)
    return entry_code, pytest_code, pytest_log[:4000], entry_log[:4000], runtime_ms


async def run_repo_sandbox(
    root: str | Path,
    files: dict[str, str] | None,
    tasks: list[ExtendFileTask],
    timeout: int = 120,
    entrypoint: str = "",
) -> tuple[SandboxResult, list[dict]]:
    """Validate planned changes on a copy of the repo subset.

    Returns (SandboxResult, changes-as-dicts). The real repo is never touched.
    """
    if not root:
        return SandboxResult(passed=False, pytest_log="no repo root", returncode=1, runtime_ms=0), []
    root_p = Path(root).expanduser()
    if not root_p.exists() or not root_p.is_dir():
        return SandboxResult(passed=False, pytest_log=f"repo root not found: {root}", returncode=1, runtime_ms=0), []

    try:
        with tempfile.TemporaryDirectory(prefix="prodrag_repo_copy_") as tmpdir:
            dst = Path(tmpdir) / "repo"
            dst.mkdir()
            copied = _copy_subset(root_p, files, dst)
            changes, patch_log = apply_changes(dst, tasks)
            loop = asyncio.get_running_loop()
            entry_code, pytest_code, pytest_log, entry_log, runtime_ms = await loop.run_in_executor(
                None,
                lambda: _run_sync(dst, entrypoint, 60, timeout),
            )
            repo_log = (
                f"copied {len(copied)} files; {patch_log}; pytest exit {pytest_code}"
                + (f"; entrypoint exit {entry_code}" if entrypoint else "")
            )
            passed = pytest_code in (0, 127) and (entry_code == 0 if entrypoint else True)
            if "timeout" in (pytest_log + entry_log).lower():
                passed = False
            result = SandboxResult(
                passed=passed,
                pytest_log=pytest_log,
                demo_log=entry_log,
                returncode=entry_code if entrypoint else pytest_code,
                runtime_ms=runtime_ms,
                repo_log=repo_log[:4000],
            )
            return result, changes
    except Exception as e:
        logger.exception("repo sandbox crashed")
        return SandboxResult(
            passed=False,
            pytest_log=f"repo sandbox error: {e}"[:4000],
            returncode=1,
            runtime_ms=0,
        ), []
