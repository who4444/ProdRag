from fastapi import APIRouter, Query

from .. import memory
from ..core import db

router = APIRouter(prefix="/memory")


@router.get("/search")
async def memory_search(query: str = Query(..., min_length=1), k: int = Query(default=5, ge=1, le=20)):
    results = await memory.remember_search(query, k=k)
    return {"query": query, "k": k, "results": results}


@router.get("/episodes")
async def list_episodes(limit: int = Query(default=20, ge=1, le=100)):
    return await db.episode_list(limit=limit)
