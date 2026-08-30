"""R&D endpoints: /rd/analyze (second-request) + /rd/research (streaming pipeline)."""

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..rd.analyzer import analyze_idea
from ..rd.pipeline import rd_research
from ..rd.schemas import AnalyzeRequest, AnalyzeResponse, RDResearchRequest

router = APIRouter(prefix="/rd")


@router.post("/analyze")
async def rd_analyze(request: Request, req: AnalyzeRequest):
    result = await analyze_idea(request.app.state.client, req.idea, req.answers)
    return JSONResponse(
        AnalyzeResponse(
            ready=result["ready"],
            requirements=result["requirements"],
            questions=result["questions"],
            draft=result["draft"],
            session_id=req.session_id,
        ).model_dump(exclude_none=True)
    )


@router.post("/research")
async def rd_research_endpoint(request: Request, req: RDResearchRequest):
    async def _gen():
        async for evt in rd_research(
            request.app.state.client, req.requirements, req.session_id, req.idea, req.k, req.k_images
        ):
            yield json.dumps(evt) + "\n"

    return StreamingResponse(_gen(), media_type="application/x-ndjson")
