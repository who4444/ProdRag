"""Codebase map — deterministic global picture first, LLM enrichment second.

Static pass (no LLM): modules + symbols, import graph resolved against repo
paths, entrypoints, impacted-module reverse closure, data files.
Enrichment (one LLM call): extension points, data flow, data availability and
cleanliness — with extension-point paths validated against the filesystem.
"""

import ast
import logging
import tomllib
from pathlib import Path

from ..config import settings
from ..core.llm import chat_json
from ..core.sparse import tokens
from ..repo.walker import language_for, walk_repo
from .citation import validate_ref
from .schemas import CodebaseMap, EvidenceRef, ExtensionPoint, IdeaUnderstanding

logger = logging.getLogger(__name__)

DATA_DIR_NAMES = {"data", "fixtures", "datasets", "golden"}
DATA_EXTENSIONS = {".jsonl", ".csv", ".parquet", ".tsv"}
MAX_MODULES = 400
MAX_PROMPT_MODULES = 60
MAX_SYMBOLS = 12


def _module_aliases(path: str) -> list[str]:
    if not path.endswith(".py"):
        return []
    stem = path[:-3]
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    dotted = stem.replace("/", ".")
    parts = dotted.split(".")
    return sorted({".".join(parts[i:]) for i in range(len(parts))})


def _resolve_import(name: str, alias_map: dict[str, str]) -> str | None:
    while name:
        hit = alias_map.get(name)
        if hit:
            return hit
        if "." not in name:
            return None
        name = name.rsplit(".", 1)[0]
    return None


def _is_data_path(path: str) -> bool:
    parts = {p.lower() for p in path.split("/")[:-1]}
    if parts & DATA_DIR_NAMES:
        return True
    return Path(path).suffix.lower() in DATA_EXTENSIONS


def _parse_python(text: str) -> tuple[list[str], list[str]]:
    """Return (top-level symbols, import module names). Never raises."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [], []
    symbols: list[str] = []
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return symbols, imports


def _impacted(seed_paths: set[str], reverse: dict[str, set[str]], depth: int = 2) -> list[str]:
    seen = set(seed_paths)
    frontier = set(seed_paths)
    for _ in range(depth):
        nxt: set[str] = set()
        for p in frontier:
            nxt |= reverse.get(p, set())
        nxt -= seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return sorted(seen)


def static_map(root: Path, idea: str, repo_id: str = "") -> CodebaseMap:
    files, _ = walk_repo(root, settings.repo_max_files, settings.repo_max_file_bytes)
    alias_map: dict[str, str] = {}
    for f in files:
        for alias in _module_aliases(f.path):
            alias_map.setdefault(alias, f.path)

    keywords = set(tokens(idea))
    modules: list[dict] = []
    edges: set[tuple[str, str]] = set()
    entrypoints: set[str] = set()
    data_files: list[str] = []
    seeds: set[str] = set()

    for f in files:
        lines = f.text.splitlines()
        if _is_data_path(f.path):
            data_files.append(f.path)
        if language_for(f.path) != "python":
            modules.append({"path": f.path, "symbols": [], "loc": len(lines)})
            continue
        symbols, imports = _parse_python(f.text)
        modules.append({"path": f.path, "symbols": symbols[:MAX_SYMBOLS], "loc": len(lines)})
        if "__main__" in f.text and "__name__" in f.text:
            entrypoints.add(f.path)
        if Path(f.path).name == "main.py":
            entrypoints.add(f.path)
        for imp in imports:
            dst = _resolve_import(imp, alias_map)
            if dst and dst != f.path:
                edges.add((f.path, dst))
        hay = (f.path + " " + " ".join(symbols)).lower()
        if keywords and any(k in hay for k in keywords):
            seeds.add(f.path)

    pyproject = next((f for f in files if f.path == "pyproject.toml"), None)
    if pyproject:
        try:
            data = tomllib.loads(pyproject.text)
            for name in (data.get("project", {}).get("scripts") or {}):
                entrypoints.add(f"[script] {name}")
        except Exception:
            logger.debug("codebase_map: pyproject parse failed")

    reverse: dict[str, set[str]] = {}
    for src, dst in edges:
        reverse.setdefault(dst, set()).add(src)

    cmap = CodebaseMap(
        modules=modules[:MAX_MODULES],
        entrypoints=sorted(entrypoints),
        dependency_edges=[list(e) for e in sorted(edges)],
        impacted_modules=_impacted(seeds, reverse),
        data_available=sorted(data_files),
    )
    logger.info(
        "codebase_map static: modules=%d edges=%d entrypoints=%d impacted=%d data=%d",
        len(cmap.modules),
        len(cmap.dependency_edges),
        len(cmap.entrypoints),
        len(cmap.impacted_modules),
        len(cmap.data_available),
    )
    return cmap


def _prompt_view(cmap: CodebaseMap) -> str:
    by_path = {m["path"]: m for m in cmap.modules}
    interesting = list(dict.fromkeys(cmap.impacted_modules + cmap.entrypoints))
    lines = []
    for p in interesting[:MAX_PROMPT_MODULES]:
        m = by_path.get(p)
        if not m:
            continue
        symbols = ", ".join(m.get("symbols", [])[:MAX_SYMBOLS])
        lines.append(f"- {p} ({m.get('loc', 0)} loc){': ' + symbols if symbols else ''}")
    return "\n".join(lines) or "(no modules matched the idea keywords)"


ENRICH_SYSTEM = (
    "You are a codebase analyst. Given a repository map and an idea, output JSON:\n"
    '{"extension_points": [{"path": str, "symbol": str, "reason": str}], '
    '"data_flow": str, "data_available": [str], "data_missing": [str], '
    '"data_cleanliness": "clean|needs_cleaning|unknown"}\n'
    "Rules: extension_points must reference files from the map (2-4 best spots). "
    "data_available lists existing repo data paths; data_missing lists data that must be "
    "collected; data_cleanliness judges whether available data can be used directly. "
    "Be concise. Return JSON only."
)


async def enrich_map(
    client, cmap: CodebaseMap, understanding: IdeaUnderstanding, idea: str, root: Path, repo_id: str = ""
) -> CodebaseMap:
    """Fill extension_points/data_flow/data_* with one LLM call; never raises."""
    user = (
        f"Idea: {idea[:500]}\n"
        f"Interpretation: {understanding.interpretation[:400]}\n"
        f"Repo id: {repo_id}\n\nModules (impacted/entrypoints):\n{_prompt_view(cmap)}\n\n"
        f"Data-like paths detected: {', '.join(cmap.data_available[:30]) or 'none'}\n"
        f"Dependency edges: {len(cmap.dependency_edges)}\n\nReturn JSON."
    )
    try:
        data = await chat_json(client, ENRICH_SYSTEM, user)
        known = {m["path"] for m in cmap.modules}
        ext: list[ExtensionPoint] = []
        for item in (data.get("extension_points") or [])[:5]:
            path = str(item.get("path", "")).strip()
            if not path or path not in known:
                continue
            ext.append(
                ExtensionPoint(
                    path=path,
                    symbol=str(item.get("symbol", ""))[:80],
                    reason=str(item.get("reason", ""))[:300],
                    evidence=[validate_ref(root, EvidenceRef(path=path))],
                )
            )
        cmap.extension_points = ext
        cmap.data_flow = str(data.get("data_flow", ""))[:800]
        missing = [str(x)[:200] for x in (data.get("data_missing") or [])][:10]
        available = [str(x)[:200] for x in (data.get("data_available") or [])][:20]
        cmap.data_available = sorted(set(cmap.data_available) | set(available))
        cmap.data_missing = missing
        cleanliness = str(data.get("data_cleanliness", "unknown"))
        cmap.data_cleanliness = cleanliness if cleanliness in ("clean", "needs_cleaning", "unknown") else "unknown"
    except Exception:
        logger.exception("codebase_map enrichment failed, keeping static map")
    return cmap
