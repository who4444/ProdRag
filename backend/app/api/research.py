from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..agent import research
from ..schemas import ResearchRequest

router = APIRouter()


@router.post("/research")
async def research_endpoint(request: Request, req: ResearchRequest):
    return StreamingResponse(
        research(
            request.app.state.client,
            request.app.state.redis,
            req.question,
            req.session_id,
            req.k,
            req.k_images,
        ),
        media_type="application/x-ndjson",
    )
