"""Memory: conversation (Supabase) + episodic (Qdrant `episodes` + Supabase)."""

import time
import uuid

from .core import db, vectorstore
from .config import settings
from .core.embeddings import text as text_emb

CONV_MAX = 20


async def conv_add(redis, session_id: str | None, role: str, content: str) -> str:
    session_id = session_id or await db.conversation_create()
    await db.message_add(session_id, role, content)
    # keep the redis key in sync for fast reads, but db is the source of truth
    await redis.rpush(
        f"conv:{session_id}", f'{{"role": {role!r}, "content": {content!r}}}'
    )
    await redis.ltrim(f"conv:{session_id}", -CONV_MAX, -1)
    await redis.expire(f"conv:{session_id}", settings.memory_ttl_s)
    return session_id


async def conv_history(redis, session_id: str | None, n: int = CONV_MAX) -> list[dict]:
    if not session_id:
        return []
    return await db.messages_for(session_id, n)


async def remember_search(question: str, k: int | None = None) -> list[dict]:
    k = k or settings.memory_top_k
    vec = (await text_emb.embed_texts([question], query=True))[0]
    return await vectorstore.search_episodes(vec, k)


async def remember_save(
    session_id: str | None, question: str, answer: str, sources: list[dict]
) -> str | None:
    ep_id = str(uuid.uuid4())
    payload = {
        "session_id": session_id,
        "created_at": time.time(),
        "question": question,
        "answer": answer,
        "sources": sources[:10],
    }
    vec = (await text_emb.embed_texts([f"Q: {question}\nA: {answer}"]))[0]
    point_id = await vectorstore.upsert_episodes([(payload, vec)])
    await db.episode_upsert(
        {"id": point_id or ep_id, "session_id": session_id, "question": question,
         "answer": answer, "sources": sources[:10]}
    )
    return point_id
