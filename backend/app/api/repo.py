"""Repo endpoints: index a local codebase, list/delete indexed repos."""

import logging

from fastapi import APIRouter, HTTPException

from ..core import db, vectorstore
from ..repo.indexer import index_repo
from ..repo.schemas import RepoIndexRequest, RepoSearchRequest
from ..tools.code_rag import code_search, code_source_items

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/repo")


@router.get("/list")
async def list_repos(limit: int = 50):
    try:
        return await db.repo_list(limit=limit)
    except Exception:
        logger.exception("repo_list failed")
        return []


@router.post("/index")
async def repo_index(req: RepoIndexRequest):
    try:
        stats = await index_repo(req.path, repo_id=req.repo_id, name=req.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return stats.model_dump()


@router.post("/search")
async def repo_search(req: RepoSearchRequest):
    result = await code_search(req.query, k=req.k, repo_id=req.repo_id, path_prefix=req.path_prefix)
    return {"hits": code_source_items(result["hits"])}


@router.get("/{repo_id}")
async def get_repo(repo_id: str):
    repo = await db.repo_get(repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="repo not indexed")
    return repo


@router.delete("/{repo_id}")
async def delete_repo(repo_id: str):
    await vectorstore.delete_code_by_repo(repo_id)
    await db.repo_delete(repo_id)
    return {"deleted": repo_id}
