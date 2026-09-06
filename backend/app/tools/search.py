"""Search tools — web search via Tavily / Serper / generic endpoint.

All providers are optional. When no key/URL is configured, web_search is a no-op
returning [] so the pipeline stays offline-first (RAG-only).

ponytail: one function, three tiny providers, httpx + retry, no new deps.
"""

import logging

import httpx

from ..config import settings
from ..core.retry import retry_async

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return bool(
        (settings.tavily_api_key or "").strip()
        or (settings.serper_api_key or "").strip()
        or (settings.search_service_url or "").strip()
    )


def _is_transient(exc: Exception) -> bool:
    from httpx import HTTPStatusError, TransportError

    if isinstance(exc, HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, TransportError)


async def _tavily_search(query: str, k: int) -> list[dict]:
    async def _call():
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": settings.tavily_api_key, "query": query, "max_results": k, "search_depth": "basic"},
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results") or []
            return [
                {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")[:400]}
                for r in results[:k]
            ]

    return await retry_async(_call, is_transient=_is_transient, label="tavily_search")


async def _serper_search(query: str, k: int) -> list[dict]:
    async def _call():
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": k},
                headers={"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            organic = data.get("organic") or []
            return [
                {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")[:400]}
                for r in organic[:k]
            ]

    return await retry_async(_call, is_transient=_is_transient, label="serper_search")


async def _generic_search(query: str, k: int) -> list[dict]:
    url = settings.search_service_url.rstrip("/")
    headers = {}
    if settings.search_service_token:
        headers["Authorization"] = f"Bearer {settings.search_service_token}"

    async def _call():
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json={"query": query, "k": k}, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # accept {results: [...]} or [...] directly
            results = data.get("results") if isinstance(data, dict) and "results" in data else data
            if not isinstance(results, list):
                return []
            return [
                {
                    "title": r.get("title", r.get("name", "")),
                    "url": r.get("url", r.get("link", "")),
                    "snippet": r.get("snippet", r.get("content", r.get("text", "")))[:400],
                }
                for r in results[:k]
            ]

    return await retry_async(_call, is_transient=_is_transient, label="search_service")


async def web_search(query: str, k: int | None = None) -> list[dict]:
    """Search the web. Returns [] when not configured or on failure (never raises)."""
    k = k or settings.search_max_results
    if not enabled() or not query.strip():
        return []
    try:
        if (settings.tavily_api_key or "").strip():
            return await _tavily_search(query, k)
        if (settings.serper_api_key or "").strip():
            return await _serper_search(query, k)
        return await _generic_search(query, k)
    except Exception:
        logger.exception("web_search failed for query=%r", query[:120])
        return []


# Tool spec for LLM function-calling (OpenAI format) — reused by agents
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for recent papers, docs, or code examples relevant to a focused query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Focused search query"},
                "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
        },
    },
}


def format_results(results: list[dict]) -> str:
    if not results:
        return "No web results."
    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or r.get("url", "")
        snippet = r.get("snippet", "")[:300]
        url = r.get("url", "")
        parts.append(f"[{i}] {title} — {snippet} ({url})")
    return "\n\n".join(parts)
