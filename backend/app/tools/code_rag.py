"""Packaged code-RAG tool — hybrid search over indexed repos.

Same retrieval shape as tools/rag.py (dense + BM25 RRF, optional rerank), but
scoped to the `code_chunks` collection and repo/path filters.
"""

import logging
import math
from collections import Counter
from pathlib import Path

from ..config import settings
from ..core import vectorstore
from ..core.embeddings import text as text_emb
from ..core.rerank import rerank
from ..core.sparse import to_sparse, tokens

logger = logging.getLogger(__name__)

RERANK_CODE_LIMIT = 1500

# Path tokens every file shares; matching them is not evidence of relevance.
_PATH_BOILERPLATE = {
    "backend", "app", "src", "lib", "py", "pyi", "python", "init", "main",
    "test", "tests", "docs", "code", "core", "utils", "files",
}

_TEST_QUERY_WORDS = ("test", "pytest", "spec", "conftest", "fixture")


def _is_test_path(path: str) -> bool:
    p = str(path).lower()
    return (
        "/tests/" in p
        or p.startswith("tests/")
        or "/test_" in p
        or Path(p).name.startswith("test_")
        or Path(p).name == "conftest.py"
    )


def _wants_tests(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in _TEST_QUERY_WORDS)


def _lexical_rerank(query: str, hits: list[dict]) -> list[dict]:
    """Offline reranker: query-term coverage + log-tf density over path/symbol/text.

    The cross-encoder reranker is a remote service (unavailable offline), and
    BM25 alone favors long prose chunks. Coverage first ("does this chunk mention
    what I asked about?") then density, with path/symbol tokens included.
    """
    q_terms = set(tokens(query))
    if not q_terms:
        return hits
    for h in hits:
        doc = f"{h.get('path', '')} {h.get('symbol', '')} {h.get('text', '')}"
        counts = Counter(tokens(doc))
        covered = [t for t in q_terms if t in counts]
        coverage = len(covered) / len(q_terms)
        # BM25 tf saturation (k1=1.2) over sqrt-length: repeated prose terms stop
        # mattering after a few occurrences, short focused code chunks win.
        density = sum(counts[t] / (counts[t] + 1.2) for t in covered) / math.sqrt(max(sum(counts.values()), 1))
        # File/symbol names are strong signals ("retrieval" -> rag/retrieval.py).
        # Offline has no dense model, so prose docs that merely mention the same
        # terms outrank the implementation unless we explicitly prefer code and
        # treat a filename/symbol hit as near-decisive.
        path = str(h.get("path", "")).lower()
        path_terms = set(tokens(f"{path} {h.get('symbol', '')}")) - _PATH_BOILERPLATE
        matched = path_terms & q_terms
        is_code = path.endswith((".py", ".pyi"))
        is_doc = path.endswith((".md", ".rst", ".txt"))
        source_bonus = 0.15 if is_code else (-0.1 if is_doc else 0.0)
        if matched:
            source_bonus += 0.6
        path_coverage = len(matched) / len(q_terms)
        h["score"] = round(coverage + 0.3 * density + 0.4 * path_coverage + source_bonus, 4)
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


async def code_search(
    question: str,
    k: int = 6,
    repo_id: str | None = None,
    path_prefix: str | None = None,
) -> dict:
    """Return {"hits": [{path, symbol, kind, start_line, end_line, text, score}]}."""
    # Offline (lexical-only) needs a wider candidate pool: BM25 without a strong
    # dense signal is noisier, so give the reranker more to work with.
    first_k = max(settings.retrieval_hybrid_k, k * (6 if settings.offline else 3))
    vec = (await text_emb.embed_texts([question], query=True))[0]
    hits = await vectorstore.search_code(
        vec,
        first_k,
        repo_id=repo_id,
        sparse_query=to_sparse(question),
        prefetch_k=first_k,
        path_prefix=path_prefix,
    )
    # Test files quote every symbol they cover; for implementation questions they
    # crowd out the source files. Drop them unless the question is about tests.
    if not _wants_tests(question):
        filtered = [h for h in hits if not _is_test_path(h.get("path", ""))]
        if filtered:
            hits = filtered

    if settings.offline:
        hits = _lexical_rerank(question, hits)
        return {"hits": hits[:k]}
    try:
        scores = await rerank(question, [h.get("text", "")[:RERANK_CODE_LIMIT] for h in hits])
    except Exception:
        logger.warning("code rerank failed for %r — falling back to RRF", question[:80], exc_info=True)
        scores = None
    if scores is not None:
        for h, s in zip(hits, scores):
            h["score"] = round(s, 3)
        hits.sort(key=lambda h: h["score"], reverse=True)
    return {"hits": hits[:k]}


def code_source_items(hits: list[dict]) -> list[dict]:
    return [
        {
            "path": h["path"],
            "symbol": h.get("symbol", ""),
            "kind": h.get("kind", ""),
            "start_line": h.get("start_line"),
            "end_line": h.get("end_line"),
            "score": round(h.get("score", 0), 3),
            "snippet": h.get("text", "")[:200],
        }
        for h in hits
    ]
