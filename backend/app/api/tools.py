"""Direct search tool endpoint for debugging — POST /tools/search.

Lets the simple frontend test web_search without running the full R&D pipeline.
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel

from ..tools.search import enabled, format_results, web_search

router = APIRouter(prefix="/tools")


class SearchRequest(BaseModel):
    query: str
    k: int = 5


@router.post("/search")
async def search_post(req: SearchRequest):
    results = await web_search(req.query, k=req.k)
    return {"enabled": enabled(), "results": results, "formatted": format_results(results)}


@router.get("/search")
async def search_get(query: str = Query(..., min_length=1), k: int = Query(default=5, ge=1, le=10)):
    """GET variant for browser/curl probing — avoids 405 when you hit the URL directly."""
    results = await web_search(query, k=k)
    return {"enabled": enabled(), "results": results, "formatted": format_results(results)}


@router.get("/search/status")
async def search_status():
    return {"enabled": enabled()}
