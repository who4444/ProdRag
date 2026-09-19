"""Deterministic, capped filesystem walker for repo indexing.

Skips VCS/dependency/cache dirs and all dotfiles (keeps `.env` secrets out of
the index). Only known text extensions are read; oversized and binary-looking
files are counted as skipped, never fatal.
"""

import logging
import os
from pathlib import Path

from .schemas import RepoFile

logger = logging.getLogger(__name__)

DEFAULT_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".tox",
    ".nox",
    ".idea",
    ".vscode",
    "site-packages",
    ".egg-info",
}

TEXT_EXTENSIONS = {
    ".py",
    ".pyi",
    ".md",
    ".rst",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".sql",
    ".sh",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".txt",
}

_LANGUAGE_BY_EXT = {
    ".py": "python",
    ".pyi": "python",
    ".md": "markdown",
    ".rst": "rst",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".ini": "ini",
    ".cfg": "ini",
    ".sql": "sql",
    ".sh": "shell",
    ".json": "json",
    ".jsonl": "jsonl",
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "text",
}


def language_for(path: str) -> str:
    return _LANGUAGE_BY_EXT.get(Path(path).suffix.lower(), "text")


def walk_repo(root: Path, max_files: int = 2000, max_file_bytes: int = 200_000) -> tuple[list[RepoFile], int]:
    """Return (files sorted by repo-relative path, skipped_count)."""
    root = Path(root)
    files: list[RepoFile] = []
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in DEFAULT_IGNORE_DIRS and not d.startswith("."))
        for fname in sorted(filenames):
            if fname.startswith("."):
                skipped += 1
                continue
            if Path(fname).suffix.lower() not in TEXT_EXTENSIONS:
                skipped += 1
                continue
            p = Path(dirpath) / fname
            try:
                if p.stat().st_size > max_file_bytes:
                    skipped += 1
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.debug("repo walker: unreadable file %s", p)
                skipped += 1
                continue
            if "\x00" in text[:1024]:
                skipped += 1
                continue
            files.append(RepoFile(path=p.relative_to(root).as_posix(), text=text))
            if len(files) >= max_files:
                logger.info("repo walker: file cap %d reached at %s", max_files, root)
                files.sort(key=lambda f: f.path)
                return files, skipped
    files.sort(key=lambda f: f.path)
    return files, skipped
