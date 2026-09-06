import pytest

from app.tools.search import WEB_SEARCH_TOOL, enabled, format_results, web_search


def test_enabled_false_when_unconfigured(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", "")
    monkeypatch.setattr(settings, "serper_api_key", "")
    monkeypatch.setattr(settings, "search_service_url", "")
    assert enabled() is False


def test_enabled_true_when_tavily_set(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", "tvly-xxx")
    assert enabled() is True


@pytest.mark.asyncio
async def test_web_search_noop_when_disabled(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", "")
    monkeypatch.setattr(settings, "serper_api_key", "")
    monkeypatch.setattr(settings, "search_service_url", "")
    res = await web_search("hello world")
    assert res == []


@pytest.mark.asyncio
async def test_web_search_noop_on_empty_query(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "tavily_api_key", "tvly-xxx")
    res = await web_search("   ")
    assert res == []


def test_format_results_empty():
    assert format_results([]) == "No web results."


def test_format_results_shape():
    out = format_results([{"title": "T", "url": "https://example.com", "snippet": "hello"}])
    assert "T" in out and "https://example.com" in out


def test_web_search_tool_spec():
    assert WEB_SEARCH_TOOL["function"]["name"] == "web_search"
    assert "query" in WEB_SEARCH_TOOL["function"]["parameters"]["properties"]
