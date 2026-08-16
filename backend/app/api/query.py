from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..rag.answer import answer
from ..rag.retrieval import search_kb
from ..schemas import QueryRequest

router = APIRouter()


@router.post("/query")
async def query_endpoint(request: Request, req: QueryRequest):
    text_hits, image_hits = await search_kb(req.question, req.k, req.k_images)
    return StreamingResponse(
        answer(request.app.state.client, req.question, text_hits, image_hits),
        media_type="application/x-ndjson",
    )
