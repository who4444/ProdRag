"""AST-aware chunker: coverage, symbols, imports, splits, fallbacks (M1)."""

from app.repo.chunker import chunk_file, chunk_python
from app.repo.schemas import RepoFile


def _source() -> str:
    big_lines = "\n".join(f"    total += {i}" for i in range(1, 140))
    return (
        '"""Module docstring."""\n'
        "import os\n"
        "from typing import Iterable\n"
        "\n"
        "X = 1\n"
        "\n"
        "def small(a):\n"
        "    return a + 1\n"
        "\n"
        "def big():\n"
        "    total = 0\n"
        f"{big_lines}\n"
        "    return total\n"
        "\n"
        "class Small:\n"
        '    """doc"""\n'
        "    value = 1\n"
        "\n"
        "    def method(self):\n"
        "        return self.value\n"
        "\n"
        "class Big:\n"
        '    """doc"""\n'
        "\n"
        "    def m1(self):\n"
        "        return 1\n"
        "\n"
        "    attr = 2\n"
        "\n"
        "    def m2(self):\n"
        "        return 2\n"
        "\n"
        "TAIL = 3\n"
    )


def test_chunk_python_covers_every_nonblank_line():
    src = _source()
    chunks = chunk_python(src, "pkg/mod.py", max_lines=80, overlap=10)
    lines = src.splitlines()

    assert chunks
    for i, line in enumerate(lines, start=1):
        if line.strip():
            assert any(c.start_line <= i <= c.end_line for c in chunks), f"line {i} uncovered: {line!r}"


def test_chunk_python_text_matches_source_slice():
    src = _source()
    chunks = chunk_python(src, "pkg/mod.py", max_lines=80, overlap=10)
    lines = src.splitlines()
    for c in chunks:
        assert c.text == "\n".join(lines[c.start_line - 1 : c.end_line])


def test_chunk_python_symbols_and_imports():
    chunks = chunk_python(_source(), "pkg/mod.py", max_lines=80, overlap=10)
    symbols = {c.symbol for c in chunks}
    assert {"small", "big", "Small", "Big"} <= symbols
    assert all(c.imports == ["os", "typing"] for c in chunks if c.language == "python")
    kinds = {c.symbol: c.kind for c in chunks if c.symbol in {"small", "Small", "Big", "big"}}
    assert kinds == {"small": "function", "big": "function", "Small": "class", "Big": "class"}


def test_large_class_splits_into_methods():
    chunks = chunk_python(_source(), "pkg/mod.py", max_lines=8, overlap=2)
    symbols = {c.symbol for c in chunks}
    assert {"Big", "Big.m1", "Big.m2"} <= symbols
    methods = [c for c in chunks if c.symbol == "Big.m1"]
    assert methods and methods[0].kind == "method"


def test_large_function_is_split_with_overlap():
    chunks = chunk_python(_source(), "pkg/mod.py", max_lines=40, overlap=5)
    big = [c for c in chunks if c.symbol == "big"]
    assert len(big) > 1
    for a, b in zip(big, big[1:]):
        assert b.start_line <= a.end_line + 1  # contiguous or overlapping
        assert b.start_line - a.end_line <= 1


def test_empty_file_yields_no_chunks():
    assert chunk_python("", "empty.py", max_lines=80, overlap=10) == []


def test_syntax_error_falls_back_to_blocks_without_raising():
    src = "def broken(:\n    pass\n" * 10
    chunks = chunk_python(src, "broken.py", max_lines=80, overlap=10)
    assert chunks
    assert all(c.kind == "block" for c in chunks)


def test_non_python_falls_back_to_line_windows():
    src = "\n".join(f"line {i}" for i in range(1, 201))
    chunks = chunk_file(RepoFile(path="docs/guide.md", text=src), max_lines=80, overlap=10)
    assert len(chunks) >= 3
    assert all(c.kind == "block" and c.language == "markdown" for c in chunks)
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 200
