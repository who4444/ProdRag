"""Repo walker: ignore rules, caps, determinism (M1)."""

from pathlib import Path

from app.repo.walker import language_for, walk_repo


def _build_repo(root: Path) -> Path:
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "mod.py").write_text("x = 1\n")
    (root / "README.md").write_text("# hi\n")
    (root / "query.sql").write_text("select 1;\n")
    (root / ".env").write_text("PRODRAG_API_TOKEN=super-secret\n")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n")
    (root / ".venv").mkdir()
    (root / ".venv" / "lib.py").write_text("bad\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "mod.pyc").write_text("junk\n")
    (root / "big.py").write_text("x = 1\n" * 40000)
    (root / "image.png").write_bytes(b"\x89PNG")
    (root / "binary.py").write_bytes(b"\x00\x01\x02\x03")
    return root


def test_walker_returns_only_indexable_text_files(tmp_path):
    repo = _build_repo(tmp_path)
    files, skipped = walk_repo(repo)
    paths = [f.path for f in files]

    assert paths == ["README.md", "pkg/__init__.py", "pkg/mod.py", "query.sql"]
    assert skipped >= 4  # .env, big.py, image.png, binary.py (ignored dirs are not visited)
    assert all("secret" not in f.text for f in files)


def test_walker_is_deterministic(tmp_path):
    repo = _build_repo(tmp_path)
    first, _ = walk_repo(repo)
    second, _ = walk_repo(repo)
    assert [f.path for f in first] == [f.path for f in second]
    assert [f.text for f in first] == [f.text for f in second]


def test_walker_respects_max_files_and_size_caps(tmp_path):
    repo = _build_repo(tmp_path)
    capped, _ = walk_repo(repo, max_files=2)
    assert len(capped) == 2

    tiny, _ = walk_repo(repo, max_file_bytes=5)
    assert "README.md" in [f.path for f in tiny]
    assert "pkg/mod.py" not in [f.path for f in tiny]


def test_language_for():
    assert language_for("a/b.py") == "python"
    assert language_for("a/b.pyi") == "python"
    assert language_for("a/b.md") == "markdown"
    assert language_for("a/b.unknown") == "text"
