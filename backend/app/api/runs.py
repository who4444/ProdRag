"""Run endpoints — start/resume the LangGraph control layer (NDJSON)."""

import json
import logging

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from ..core import db
from ..feasibility.pipeline import repo_root
from ..runs.controller import get_run_state, resume_run, start_run
from ..runs.schemas import ResumeRequest, RunRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs")


@router.get("")
async def list_runs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    repo_id: str | None = None,
    status: str | None = None,
):
    try:
        return await db.rd_run_list(limit=limit, offset=offset, repo_id=repo_id, status=status)
    except Exception:
        logger.exception("rd_run_list failed")
        return []


@router.get("/{run_id}")
async def get_run(run_id: str):
    run = await db.rd_run_get(run_id)
    state = await get_run_state(run_id)
    events = await db.rd_events_for(run_id)
    if not run and not state.get("values"):
        raise HTTPException(status_code=404, detail="run not found")
    return {"run": run, "state": state, "events": events}


@router.post("")
async def create_run(request: Request, req: RunRequest):
    try:
        await repo_root(req.repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def _gen():
        async for evt in start_run(request.app.state.client, req):
            yield json.dumps(evt) + "\n"

    return StreamingResponse(_gen(), media_type="application/x-ndjson")


@router.post("/{run_id}/resume")
async def resume(request: Request, run_id: str, req: ResumeRequest):
    run = await db.rd_run_get(run_id)
    state = await get_run_state(run_id)
    if not run and not state.get("values"):
        raise HTTPException(status_code=404, detail="run not found")
    if not state.get("interrupts"):
        raise HTTPException(status_code=409, detail="run is not waiting for input")

    if req.answers is not None:
        value = req.answers
    elif req.accept is not None:
        value = {"accept": req.accept, "idea": req.idea}
    else:
        raise HTTPException(status_code=422, detail="provide answers or accept")

    async def _gen():
        async for evt in resume_run(request.app.state.client, run_id, value):
            yield json.dumps(evt) + "\n"

    return StreamingResponse(_gen(), media_type="application/x-ndjson")
