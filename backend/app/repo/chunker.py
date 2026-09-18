"""AST-aware chunking for repo indexing.

Python files are split on top-level definitions (functions/classes); large
class bodies split into method chunks. Gaps between definitions are emitted as
`block` chunks so every non-blank line is covered exactly once. Oversized spans
are split into overlapping line windows. Non-Python files fall back to line
windows. Never raises — a syntax-error Python file becomes block chunks.
"""

import ast

from .schemas import CodeChunk, RepoFile
from .walker import language_for


def _node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        return min(d.lineno for d in decorators)
    return node.lineno


def _imports(tree: ast.AST) -> list[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return sorted(found)


def _window_chunks(
    lines: list[str],
    start: int,
    end: int,
    path: str,
    symbol: str,
    kind: str,
    language: str,
    max_lines: int,
    overlap: int,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    pos = start
    while pos <= end:
        stop = min(pos + max_lines - 1, end)
        text = "\n".join(lines[pos - 1 : stop])
        if text.strip():
            chunks.append(
                CodeChunk(
                    path=path,
                    language=language,
                    symbol=symbol,
                    kind=kind,
                    start_line=pos,
                    end_line=stop,
                    text=text,
                )
            )
        if stop == end:
            break
        nxt = stop + 1 - overlap
        pos = nxt if nxt > pos else stop + 1
    return chunks


def chunk_python(text: str, path: str, max_lines: int, overlap: int) -> list[CodeChunk]:
    lines = text.splitlines()
    if not lines:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _window_chunks(lines, 1, len(lines), path, "", "block", "python", max_lines, overlap)

    spans: list[tuple[int, int, str, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((_node_start(node), node.end_lineno, node.name, "function"))
        elif isinstance(node, ast.ClassDef):
            start, end = _node_start(node), node.end_lineno
            if end - start + 1 <= max_lines:
                spans.append((start, end, node.name, "class"))
                continue
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if methods:
                header_end = _node_start(methods[0]) - 1
                if header_end >= start:
                    spans.append((start, header_end, node.name, "class"))
                spans.extend((_node_start(m), m.end_lineno, f"{node.name}.{m.name}", "method") for m in methods)
            else:
                spans.append((start, end, node.name, "class"))
    spans.sort()

    chunks: list[CodeChunk] = []
    cursor = 1
    for start, end, symbol, kind in spans:
        start = max(start, cursor)
        if start > end:
            continue
        if start > cursor:
            chunks += _window_chunks(lines, cursor, start - 1, path, "", "block", "python", max_lines, overlap)
        chunks += _window_chunks(lines, start, end, path, symbol, kind, "python", max_lines, overlap)
        cursor = end + 1
    if cursor <= len(lines):
        chunks += _window_chunks(lines, cursor, len(lines), path, "", "block", "python", max_lines, overlap)

    imports = _imports(tree)
    for chunk in chunks:
        chunk.imports = imports
    return chunks


def chunk_file(repo_file: RepoFile, max_lines: int = 80, overlap: int = 10) -> list[CodeChunk]:
    language = language_for(repo_file.path)
    if language == "python":
        return chunk_python(repo_file.text, repo_file.path, max_lines, overlap)
    lines = repo_file.text.splitlines()
    if not lines:
        return []
    return _window_chunks(lines, 1, len(lines), repo_file.path, "", "block", language, max_lines, overlap)
