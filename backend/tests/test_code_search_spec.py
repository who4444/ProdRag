"""code-RAG tool: hybrid args, rerank fallback, source items (M1)."""

import pytest

from app.tools.code_rag import code_search, code_source_items


def _hits() -> list[dict]:
    return [
        {
            "path": "svc/router.py",
            "symbol": "route_payments",
            "kind": "function",
            "start_line": 4,
            "end_line": 8,
            "text": "def route_payments(order): ...",
            "score": 0.4,
        },
        {
            "path": "svc/payments.py",
            "symbol": "charge",
            "kind": "function",
            "start_line": 1,
            "end_line": 3,
            "text": "def charge(order): ...",
            "score": 0.2,
        },
    ]


@pytest.mark.asyncio
async def test_code_search_passes_filters_and_reranks(monkeypatch):
    captured: dict = {}

    async def _search(vec, k, repo_id=None, sparse_query=None, prefetch_k=None, path_prefix=None):
        captured.update(k=k, repo_id=repo_id, sparse_query=sparse_query, prefetch_k=prefetch_k, path_prefix=path_prefix)
        return [dict(h) for h in _hits()]

    async def _embed(texts, query=False):
        captured["embedded_query"] = texts[0]
        return [[0.1] * 4 for _ in texts]

    async def _rerank(question, texts):
        return [0.1, 0.9]  # payments.py wins

    monkeypatch.setattr("app.tools.code_rag.vectorstore.search_code", _search)
    monkeypatch.setattr("app.tools.code_rag.text_emb.embed_texts", _embed)
    monkeypatch.setattr("app.tools.code_rag.rerank", _rerank)

    result = await code_search("where is the charge function", k=2, repo_id="r1", path_prefix="svc/")

    assert captured["repo_id"] == "r1"
    assert captured["path_prefix"] == "svc/"
    assert captured["sparse_query"]["indices"]
    assert captured["prefetch_k"] >= 2
    assert captured["embedded_query"] == "where is the charge function"
    assert result["hits"][0]["path"] == "svc/payments.py"


@pytest.mark.asyncio
async def test_code_search_rerank_failure_keeps_rrf_order(monkeypatch):
    async def _search(vec, k, repo_id=None, sparse_query=None, prefetch_k=None, path_prefix=None):
        return [dict(h) for h in _hits()]

    async def _embed(texts, query=False):
        return [[0.1] * 4 for _ in texts]

    async def _rerank(question, texts):
        raise RuntimeError("reranker down")

    monkeypatch.setattr("app.tools.code_rag.vectorstore.search_code", _search)
    monkeypatch.setattr("app.tools.code_rag.text_emb.embed_texts", _embed)
    monkeypatch.setattr("app.tools.code_rag.rerank", _rerank)

    result = await code_search("router", k=2)
    assert [h["path"] for h in result["hits"]] == ["svc/router.py", "svc/payments.py"]


@pytest.mark.asyncio
async def test_code_search_offline_uses_lexical_coverage_rerank(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "offline", True)
    long_prose = "sandbox run subprocess " * 40
    hits = [
        {"path": "PROJECT.md", "symbol": "", "kind": "block", "start_line": 1, "end_line": 5, "text": long_prose, "score": 99.0},
        {
            "path": "backend/app/code/sandbox.py",
            "symbol": "run_sandbox",
            "kind": "function",
            "start_line": 183,
            "end_line": 225,
            "text": "def run_sandbox(files):\n    subprocess.run([sys.executable, 'demo.py'])\n",
            "score": 1.0,
        },
        {"path": "docs/other.md", "symbol": "", "kind": "block", "start_line": 1, "end_line": 3, "text": "unrelated text", "score": 0.5},
    ]

    async def _search(vec, k, repo_id=None, sparse_query=None, prefetch_k=None, path_prefix=None):
        return [dict(h) for h in hits]

    async def _embed(texts, query=False):
        return [[0.1] * 4 for _ in texts]

    async def _rerank_should_not_run(question, texts):
        raise AssertionError("remote reranker must not be called in offline mode")

    monkeypatch.setattr("app.tools.code_rag.vectorstore.search_code", _search)
    monkeypatch.setattr("app.tools.code_rag.text_emb.embed_texts", _embed)
    monkeypatch.setattr("app.tools.code_rag.rerank", _rerank_should_not_run)

    result = await code_search("where is the sandbox run and subprocess", k=3)
    assert result["hits"][0]["path"] == "backend/app/code/sandbox.py"
    assert result["hits"][0]["score"] > result["hits"][-1]["score"]


@pytest.mark.asyncio
async def test_offline_rerank_prefers_code_over_docs(monkeypatch):
    """Prose that repeats query terms must not outrank the implementation."""
    from app.config import settings

    monkeypatch.setattr(settings, "offline", True)
    prose = ("rag_search cache retrieval embedding " * 30).strip()
    hits = [
        {"path": "PROJECT.md", "symbol": "", "kind": "block", "start_line": 1, "end_line": 80, "text": prose, "score": 9.0},
        {
            "path": "backend/app/tools/rag.py",
            "symbol": "rag_search",
            "kind": "function",
            "start_line": 11,
            "end_line": 14,
            "text": "async def rag_search(question, k=4):\n    return await search_kb(question, k)",
            "score": 1.0,
        },
    ]

    async def _search(vec, k, repo_id=None, sparse_query=None, prefetch_k=None, path_prefix=None):
        return [dict(h) for h in hits]

    async def _embed(texts, query=False):
        return [[0.1] * 4 for _ in texts]

    monkeypatch.setattr("app.tools.code_rag.vectorstore.search_code", _search)
    monkeypatch.setattr("app.tools.code_rag.text_emb.embed_texts", _embed)

    result = await code_search("rag_search cache retrieval embedding", k=2)
    assert result["hits"][0]["path"] == "backend/app/tools/rag.py"


@pytest.mark.asyncio
async def test_code_search_drops_test_files_unless_asked(monkeypatch):
    """Implementation questions must not be answered with test files."""
    hits = [
        {"path": "backend/tests/test_rag.py", "symbol": "test_rag_search", "kind": "function", "start_line": 1, "end_line": 3, "text": "def test_rag_search(): ...", "score": 9.0},
        {"path": "backend/app/tools/rag.py", "symbol": "rag_search", "kind": "function", "start_line": 11, "end_line": 14, "text": "async def rag_search(...): ...", "score": 1.0},
    ]

    async def _search(vec, k, repo_id=None, sparse_query=None, prefetch_k=None, path_prefix=None):
        return [dict(h) for h in hits]

    async def _embed(texts, query=False):
        return [[0.1] * 4 for _ in texts]

    async def _rerank(question, texts):
        return [0.9] + [0.1] * (len(texts) - 1)

    monkeypatch.setattr("app.tools.code_rag.vectorstore.search_code", _search)
    monkeypatch.setattr("app.tools.code_rag.text_emb.embed_texts", _embed)
    monkeypatch.setattr("app.tools.code_rag.rerank", _rerank)

    impl = await code_search("where is rag_search implemented", k=2)
    assert [h["path"] for h in impl["hits"]] == ["backend/app/tools/rag.py"]

    tests = await code_search("which pytest covers rag_search", k=2)
    assert any(h["path"] == "backend/tests/test_rag.py" for h in tests["hits"])


def test_code_source_items_shape():
    items = code_source_items(_hits())
    assert set(items[0]) == {"path", "symbol", "kind", "start_line", "end_line", "score", "snippet"}
    assert items[0]["path"] == "svc/router.py"
    assert items[0]["start_line"] == 4
    assert len(items[0]["snippet"]) <= 200
