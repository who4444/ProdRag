from fastapi import APIRouter, HTTPException, Query

from ..core import db

router = APIRouter(prefix="/conversations")


@router.get("")
async def list_conversations(limit: int = Query(default=20, ge=1, le=100), offset: int = Query(default=0, ge=0)):
    return await db.conversation_list(limit=limit, offset=offset)


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str):
    conv = await db.conversation_get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conv


@router.get("/{conversation_id}/messages")
async def list_messages(conversation_id: str, n: int = Query(default=20, ge=1, le=100)):
    conv = await db.conversation_get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="conversation not found")
    return await db.messages_for(conversation_id, n=n)


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    conv = await db.conversation_get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="conversation not found")
    # Best-effort: delete via db (supabase will cascade messages via FK if configured, else manual)
    # We have no db.conversation_delete yet, use direct table delete via _run
    from ..core.db import _client, _run

    await _run(_client().table("conversations").delete().eq("id", conversation_id).execute)
    return {"deleted": conversation_id}
