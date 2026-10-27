"""Repo context gathering for extend mode — indexed repo via code RAG or local path.

Never raises: any failure downgrades to mode="none" with the requested path kept
on the context so tests/UI can still show something useful.
"""

import logging
import re
from pathlib import Path

from ..config import settings
from ..repo.walker import walk_repo
from .schemas import RepoContext

logger = logging.getLogger(__name__)

_MAX_FILES = 60  # files carried into the LLM context (caps prompt size)
_MAX_FILE_BYTES = 40_000  # per-file cap on the text shipped to the LLM
_MAX_TOTAL_BYTES = 240_000  # total context budget


def _snippet(text: str, start: int, end: int, ctx: int = 3) -> str:
    lines = text.splitlines()
    a = max(0, start - 1 - ctx)
    b = min(len(lines), end + ctx)
    return "\n".join(lines[a:b])


async def _hits_from_code_search(idea: str, repo_id: str, k: int) -> list[dict]:
    from ..tools.code_rag import code_search

    out = await code_search(idea, k=k, repo_id=repo_id)
    return out.get("hits") or []


def _lexical_hits(files: dict[str, str], idea: str, k: int) -> list[dict]:
    """Offline fallback: substring/score scan over a small local tree."""
    import math
    from collections import Counter

    from ..core.sparse import tokens

    q_terms = [t for t in tokens(idea) if len(t) > 2][:24]
    hits = []
    for path, text in files.items():
        low = text.lower()
        counts = Counter(tokens(text))
        covered = [t for t in q_terms if t in counts]
        coverage = len(covered) / len(q_terms) if q_terms else 0.0
        density = sum(counts[t] / (counts[t] + 1.2) for t in covered) / math.sqrt(max(sum(counts.values()), 1))
        path_terms = set(tokens(path)) - {"py", "md"}
        matched = {t for t in q_terms if t in path_terms or t in low}
        score = coverage + 0.3 * density + 0.4 * (len(matched) / len(q_terms) if q_terms else 0.0)
        if score <= 0:
            continue
        # first matching line for context
        line_no = 0
        if covered:
            for i, line in enumerate(low.splitlines(), 1):
                if any(t in line for t in covered):
                    line_no = i
                    break
        hits.append({
            "path": path,
            "start_line": line_no or 1,
            "end_line": line_no or 1,
            "text": text[:800],
            "score": round(score, 4),
        })
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k]


def _persist_entities(idea: str) -> list[str]:
    """Extract likely repo identifiers from the idea for secondary retrieval queries."""
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", idea)
    cap = [w for w in words if any(c.isupper() for c in w)]
    snake = [w for w in words if "_" in w]
    return list(dict.fromkeys(cap + snake))[:6]


async def _build_from_local(path: str, idea: str, k: int, max_files_cap: int) -> RepoContext:
    root = Path(path).expanduser()
    if not root.exists() or not root.is_dir():
        return RepoContext(repo_id="", root=path, mode="none", files={}, hits=[])
    files, _ = walk_repo(root, max_files=max_files_cap, max_file_bytes=settings.repo_max_file_bytes)
    fmap = {f.path: f.text for f in files}
    hits = _lexical_hits(fmap, idea, k)
    return RepoContext(repo_id="", root=str(root.resolve()), mode="local", files=fmap, hits=hits)


async def _build_from_indexed(repo_id: str, idea: str, k: int) -> RepoContext:
    from ..core import db
    from ..core import vectorstore
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    info = await db.repo_get(repo_id)
    root = str(info.get("root", "")) if info else ""

    hits: list[dict] = []
    try:
        hits = await _hits_from_code_search(idea, repo_id, k)
    except Exception:
        logger.exception("code_search failed for repo %s", repo_id)

    # Secondary queries for entities mentioned in the idea ("sandbox", "SandboxResult"...)
    for ent in _persist_entities(idea):
        if len(hits) >= k:
            break
        try:
            extra = await _hits_by_path_symbol(ent, repo_id, k)
            hits.extend(h for h in extra if h["path"] not in {x["path"] for x in hits})
        except Exception:
            logger.exception("secondary code_search failed for %s", ent)

    files: dict[str, str] = {}
    try:
        # Pull full texts for the hit paths directly from Qdrant payloads
        client = await vectorstore.get_client()
        unique_paths = list(dict.fromkeys(h["path"] for h in hits))[:_MAX_FILES]
        scroll, _ = await client.scroll(
            settings.collection_code,
            scroll_filter=Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))]),
            limit=1000,
            with_payload=True,
        )
        by_path: dict[str, str] = {}
        for pt in scroll:
            p = (pt.payload or {}).get("path", "")
            txt = (pt.payload or {}).get("text", "")
            if p:
                prev = by_path.get(p)
                by_path[p] = txt if prev is None else _merge_chunk(prev, txt)
        for p in unique_paths:
            if p in by_path:
                files[p] = by_path[p][:_MAX_FILE_BYTES]
    except Exception:
        logger.exception("repo files fetch failed for %s", repo_id)

    return RepoContext(repo_id=repo_id, root=root, mode="repo" if files else ("none" if not hits else "local"), files=files, hits=hits)


def _merge_chunk(prev: str, txt: str) -> str:
    if not prev:
        return txt
    if not txt or txt in prev:
        return prev
    return prev + "\n" + txt


async def _hits_by_path_symbol(term: str, repo_id: str, k: int) -> list[dict]:
    from ..tools.code_rag import code_search

    out = await code_search(term, k=k, repo_id=repo_id)
    return out.get("hits") or []


async def build_repo_context(idea: str, repo_id: str | None = None, path: str | None = None, k: int = 8) -> RepoContext:
    """Build RepoContext from an indexed repo (mode=repo) or a local path (mode=local)."""
    if repo_id:
        try:
            ctx = await _build_from_indexed(repo_id, idea, k)
            if ctx.mode != "none":
                return ctx
        except Exception:
            logger.exception("indexed repo context failed, trying local path")
    if path:
        return await _build_from_local(path, idea, k, max_files_cap=settings.repo_max_files)
    # Last resort: pure lexical none-context; tests use the direct path anyway
    return RepoContext(repo_id=repo_id or "", root=path or "", mode="none", files={}, hits=[])


def context_digest(ctx: RepoContext, max_chars: int = 6000) -> str:
    """LLM-facing digest: path map + top hits with short snippets."""
    lines: list[str] = []
    if ctx.root:
        lines.append(f"repo root: {ctx.root}")
    if ctx.files:
        lines.append("files:")
        for p in sorted(ctx.files)[:_MAX_FILES]:
            n = len(ctx.files[p].splitlines())
            lines.append(f"- {p} ({n} lines)")
    for h in ctx.hits[:8]:
        lines.append(
            f"hit {h.get('path')}:{h.get('start_line')}-{h.get('end_line')} ({h.get('score', 0)}): "
            + _snippet(str(h.get("text", "")), int(h.get("start_line", 1)), int(h.get("end_line", 1)), ctx=2)[:400]
        )
    return "\n".join(lines)[:max_chars]


def selective_files(ctx: RepoContext, spec_paths: list[str], cap_bytes: int = _MAX_TOTAL_BYTES) -> dict[str, str]:
    """Files to ship with change tasks: hit-path texts + any requested focus paths present in ctx."""
    chosen: dict[str, str] = {}
    budget = cap_bytes
    wanted = list(dict.fromkeys(spec_paths + [h.get("path", "") for h in ctx.hits[:8]]))
    for p in wanted:
        if not p or p not in ctx.files or p in chosen:
            continue
        txt = ctx.files[p]
        if len(txt) > budget:
            txt = txt[:max(0, budget)]
        if not txt:
            continue
        chosen[p] = txt
        budget -= len(txt)
        if budget <= 0:
            break
    return chosen
