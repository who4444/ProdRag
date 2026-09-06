"""Supabase Postgres access via the supabase REST client.

Tables (public schema): documents, parts, conversations, messages, episodes,
research_runs, research_artifacts, code_runs, code_artifacts.
Schema is applied out-of-band (Supabase SQL editor / psql) — see schema.sql.
"""

import asyncio
import datetime
from functools import lru_cache

from supabase import create_client

from ..config import settings


def _expires_at(ttl_s: int) -> str | None:
    if not ttl_s or ttl_s <= 0:
        return None
    return (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=ttl_s)).isoformat()


@lru_cache(maxsize=1)
def _client():
    return create_client(settings.supabase_url, settings.supabase_secret_key)


async def _run(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------

async def document_upsert(doc: dict) -> None:
    """Insert or replace a document row (id is the PK)."""
    await _run(_client().table("documents").upsert(doc).execute)


async def document_get(doc_id: str) -> dict | None:
    res = await _run(_client().table("documents").select("*").eq("id", doc_id).execute)
    return (res.data or [None])[0]


async def document_update(doc_id: str, fields: dict) -> None:
    await _run(_client().table("documents").update(fields).eq("id", doc_id).execute)


async def document_list(status: str | None = None) -> list[dict]:
    q = _client().table("documents").select("*").order("created_at", desc=True)
    if status:
        q = q.eq("status", status)
    res = await _run(q.execute)
    return res.data


async def document_by_hash(content_hash: str) -> dict | None:
    """First successfully-ingested document with this content hash (upload dedup)."""
    res = await _run(
        _client().table("documents")
        .select("*")
        .eq("content_hash", content_hash)
        .eq("status", "done")
        .limit(1)
        .execute
    )
    return (res.data or [None])[0]


async def document_delete(doc_id: str) -> None:
    """Delete the row; parts cascade via FK."""
    await _run(_client().table("documents").delete().eq("id", doc_id).execute)


# --------------------------------------------------------------------------
# parts (one row per Qdrant point)
# --------------------------------------------------------------------------

async def part_insert_many(parts: list[dict]) -> None:
    """Upsert part rows (id is the PK) — idempotent on re-ingestion."""
    if not parts:
        return
    await _run(_client().table("parts").upsert(parts).execute)


async def parts_for_document(doc_id: str) -> list[dict]:
    res = await _run(_client().table("parts").select("*").eq("document_id", doc_id).execute)
    return res.data


async def part_delete_for_document(doc_id: str) -> None:
    await _run(_client().table("parts").delete().eq("document_id", doc_id).execute)


# --------------------------------------------------------------------------
# conversations / messages
# --------------------------------------------------------------------------

async def conversation_create() -> str:
    import uuid

    cid = str(uuid.uuid4())
    expires = _expires_at(settings.conversations_ttl_s)
    payload: dict = {"id": cid}
    if expires:
        payload["expires_at"] = expires
    await _run(_client().table("conversations").insert(payload).execute)
    return cid


async def conversation_get(conversation_id: str) -> dict | None:
    res = await _run(_client().table("conversations").select("*").eq("id", conversation_id).execute)
    return (res.data or [None])[0]


async def conversation_list(limit: int = 20, offset: int = 0) -> list[dict]:
    q = _client().table("conversations").select("*").order("created_at", desc=True).limit(limit)
    res = await _run(q.execute)
    data = res.data
    if offset:
        data = data[offset:]
    return data


async def message_add(conversation_id: str, role: str, content: str) -> None:
    await _run(
        _client().table("messages")
        .insert({"conversation_id": conversation_id, "role": role, "content": content})
        .execute
    )


async def messages_for(conversation_id: str, n: int = 20) -> list[dict]:
    res = await _run(
        _client().table("messages")
        .select("role,content")
        .eq("conversation_id", conversation_id)
        .order("id", desc=True)
        .limit(n)
        .execute
    )
    return list(reversed(res.data))


# --------------------------------------------------------------------------
# episodes (episodic memory)
# --------------------------------------------------------------------------

async def episode_upsert(ep: dict) -> None:
    """Insert or replace an episode row (id is the PK, == qdrant point id)."""
    expires = _expires_at(settings.episodes_ttl_s)
    if expires:
        ep = {**ep, "expires_at": expires}
    await _run(_client().table("episodes").upsert(ep).execute)


async def episode_list(limit: int = 50) -> list[dict]:
    res = await _run(
        _client().table("episodes").select("*").order("created_at", desc=True).limit(limit).execute
    )
    return res.data


# --------------------------------------------------------------------------
# research_runs / research_artifacts (R&D pipeline)
# --------------------------------------------------------------------------

async def research_run_create(run_id: str, session_id: str | None, idea: str, requirements: dict) -> None:
    expires = _expires_at(settings.research_run_ttl_s)
    payload: dict = {"id": run_id, "session_id": session_id, "idea": idea, "requirements": requirements, "status": "running"}
    if expires:
        payload["expires_at"] = expires
    await _run(_client().table("research_runs").insert(payload).execute)


async def research_run_list(limit: int = 20, offset: int = 0, session_id: str | None = None, status: str | None = None) -> list[dict]:
    q = _client().table("research_runs").select("*").order("created_at", desc=True).limit(limit + offset)
    if session_id:
        q = q.eq("session_id", session_id)
    if status:
        q = q.eq("status", status)
    res = await _run(q.execute)
    data = res.data
    if offset:
        data = data[offset : offset + limit]
    else:
        data = data[:limit]
    return data


async def research_run_update(run_id: str, fields: dict) -> None:
    await _run(_client().table("research_runs").update(fields).eq("id", run_id).execute)


async def research_run_get(run_id: str) -> dict | None:
    res = await _run(_client().table("research_runs").select("*").eq("id", run_id).execute)
    return (res.data or [None])[0]


async def research_artifact_upsert(run_id: str, artifact: dict, validation: dict) -> None:
    await _run(
        _client().table("research_artifacts").upsert(
            {"run_id": run_id, "artifact": artifact, "validation": validation}
        ).execute
    )


async def research_artifact_get(run_id: str) -> dict | None:
    res = await _run(_client().table("research_artifacts").select("*").eq("run_id", run_id).execute)
    return (res.data or [None])[0]


# --------------------------------------------------------------------------
# code_runs / code_artifacts (code generation)
# --------------------------------------------------------------------------

async def code_run_create(run_id: str, artifact_run_id: str | None, artifact: dict, status: str = "running") -> None:
    coding_spec = artifact.get("coding_spec") if isinstance(artifact, dict) and "coding_spec" in artifact else artifact
    expires = _expires_at(settings.code_run_ttl_s)
    payload: dict = {"id": run_id, "artifact_run_id": artifact_run_id, "coding_spec": coding_spec if isinstance(coding_spec, dict) else {}, "files": [], "status": status}
    if expires:
        payload["expires_at"] = expires
    await _run(_client().table("code_runs").insert(payload).execute)


async def code_run_list(limit: int = 20, offset: int = 0, artifact_run_id: str | None = None, status: str | None = None) -> list[dict]:
    q = _client().table("code_runs").select("*").order("created_at", desc=True).limit(limit + offset)
    if artifact_run_id:
        q = q.eq("artifact_run_id", artifact_run_id)
    if status:
        q = q.eq("status", status)
    res = await _run(q.execute)
    data = res.data
    if offset:
        data = data[offset : offset + limit]
    else:
        data = data[:limit]
    return data


async def code_run_update(run_id: str, fields: dict) -> None:
    await _run(_client().table("code_runs").update(fields).eq("id", run_id).execute)


async def code_run_get(run_id: str) -> dict | None:
    res = await _run(_client().table("code_runs").select("*").eq("id", run_id).execute)
    return (res.data or [None])[0]


async def code_artifact_upsert(run_id: str, artifact: dict, sandbox: dict) -> None:
    await _run(
        _client().table("code_artifacts").upsert(
            {"run_id": run_id, "artifact": artifact, "sandbox": sandbox}
        ).execute
    )


async def code_artifact_get(run_id: str) -> dict | None:
    res = await _run(_client().table("code_artifacts").select("*").eq("run_id", run_id).execute)
    return (res.data or [None])[0]
