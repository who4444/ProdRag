"""Feasibility endpoints: /feasibility/understand (handshake) + /analyze (stream)."""

import json
import logging

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..core import db
from ..feasibility.pipeline import feasibility_analyze, repo_root
from ..feasibility.schemas import AnalyzeRequest, UnderstandRequest, UnderstandResponse
from ..feasibility.understand import understand_idea

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feasibility")


@router.get("/runs")
async def list_runs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    repo_id: str | None = None,
    status: str | None = None,
):
    try:
        return await db.feasibility_run_list(limit=limit, offset=offset, repo_id=repo_id, status=status)
    except Exception:
        logger.exception("feasibility_run_list failed")
        return []


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    run = await db.feasibility_run_get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    report = await db.feasibility_report_get(run_id)
    return {"run": run, "report": report}


@router.post("/understand")
async def feasibility_understand(request: Request, req: UnderstandRequest):
    result = await understand_idea(request.app.state.client, req.idea, req.answers)
    return JSONResponse(
        UnderstandResponse(
            ready=result["ready"],
            understanding=result["understanding"],
            questions=result["questions"],
        ).model_dump(exclude_none=True)
    )


@router.post("/analyze")
async def feasibility_analyze_endpoint(request: Request, req: AnalyzeRequest):
    try:
        root = await repo_root(req.repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def _gen():
        async for evt in feasibility_analyze(request.app.state.client, req, root):
            yield json.dumps(evt) + "\n"

    return StreamingResponse(_gen(), media_type="application/x-ndjson")
