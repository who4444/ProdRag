"""Repo indexer: idempotent re-index, batching, stats, deterministic ids (M1)."""

import pytest

from app.repo.indexer import EMBED_BATCH, index_repo, repo_id_for


def _build_repo(tmp_path):
    repo = tmp_path / "demo_repo"
    (repo / "svc").mkdir(parents=True)
    (repo / "svc" / "router.py").write_text(
        "import json\n\n\ndef route_payments(order):\n    return json.dumps(order)\n"
    )
    (repo / "README.md").write_text("# demo\n")
    return repo


def _patch_indexer(monkeypatch):
    events: list[tuple] = []
    rows: list[dict] = []

    async def _ensure():
        events.append(("ensure",))

    async def _delete(repo_id):
        events.append(("delete", repo_id))

    async def _upsert(items):
        events.append(("upsert", len(items)))
        return len(items)

    async def _embed(texts):
        events.append(("embed", len(texts)))
        return [[0.0] * 4 for _ in texts]

    async def _repo_upsert(row):
        rows.append(row)

    monkeypatch.setattr("app.repo.indexer.vectorstore.ensure_collections", _ensure)
    monkeypatch.setattr("app.repo.indexer.vectorstore.delete_code_by_repo", _delete)
    monkeypatch.setattr("app.repo.indexer.vectorstore.upsert_code", _upsert)
    monkeypatch.setattr("app.repo.indexer.text_emb.embed_texts", _embed)
    monkeypatch.setattr("app.repo.indexer.db.repo_upsert", _repo_upsert)
    return events, rows


@pytest.mark.asyncio
async def test_index_repo_deletes_first_embeds_and_records_stats(tmp_path, monkeypatch):
    repo = _build_repo(tmp_path)
    events, rows = _patch_indexer(monkeypatch)

    stats = await index_repo(str(repo))

    assert stats.repo_id == repo_id_for(repo)
    assert stats.files == 2
    assert stats.chunks >= 2
    assert stats.root == str(repo.resolve())
    kinds = [e[0] for e in events]
    assert ("delete", stats.repo_id) in events
    assert kinds.index("delete") < kinds.index("upsert")  # stale chunks dropped before upsert
    assert rows and rows[0]["id"] == stats.repo_id and rows[0]["chunks"] == stats.chunks


@pytest.mark.asyncio
async def test_index_repo_payload_fields(tmp_path, monkeypatch):
    repo = _build_repo(tmp_path)
    captured: list[dict] = []

    async def _upsert(items):
        captured.extend(p for p, _ in items)
        return len(items)

    async def _ensure():
        return None

    async def _delete(repo_id):
        return None

    async def _embed(texts):
        return [[0.0] * 4 for _ in texts]

    async def _repo_upsert(row):
        return None

    monkeypatch.setattr("app.repo.indexer.vectorstore.ensure_collections", _ensure)
    monkeypatch.setattr("app.repo.indexer.vectorstore.delete_code_by_repo", _delete)
    monkeypatch.setattr("app.repo.indexer.vectorstore.upsert_code", _upsert)
    monkeypatch.setattr("app.repo.indexer.text_emb.embed_texts", _embed)
    monkeypatch.setattr("app.repo.indexer.db.repo_upsert", _repo_upsert)

    stats = await index_repo(str(repo), name="custom name")
    fn = next(p for p in captured if p["symbol"] == "route_payments")

    assert stats.name == "custom name"
    assert fn["repo_id"] == stats.repo_id
    assert fn["path"] == "svc/router.py"
    assert fn["kind"] == "function"
    assert fn["start_line"] == 4 and fn["end_line"] == 5
    assert fn["imports"] == ["json"]
    assert "route_payments" in fn["text"]


@pytest.mark.asyncio
async def test_index_repo_batches_embeddings(tmp_path, monkeypatch):
    repo = tmp_path / "many"
    repo.mkdir()
    funcs = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(EMBED_BATCH + 5))
    (repo / "many.py").write_text(funcs)
    events, _ = _patch_indexer(monkeypatch)

    stats = await index_repo(str(repo))
    embed_calls = [e for e in events if e[0] == "embed"]

    assert stats.chunks == EMBED_BATCH + 5
    assert len(embed_calls) == 2
    assert sum(n for _, n in embed_calls) == EMBED_BATCH + 5


@pytest.mark.asyncio
async def test_index_repo_str_is_deterministic(tmp_path):
    repo = _build_repo(tmp_path)
    assert repo_id_for(repo) == repo_id_for(repo)
    assert repo_id_for(repo) != repo_id_for(repo / "svc")


@pytest.mark.asyncio
async def test_index_repo_rejects_missing_path(tmp_path):
    with pytest.raises(ValueError):
        await index_repo(str(tmp_path / "does-not-exist"))
