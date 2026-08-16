from app.agent import _format_sources


def test_format_sources_cites_text_and_figures():
    result = {
        "text": [{"page": 3, "score": 0.9, "text": "the method uses attention"}],
        "images": [{"page": 5, "object_key": "docs/x/figures/5_0.png"}],
    }
    out = _format_sources(result)
    assert "[1]" in out and "attention" in out
    assert "object_key=docs/x/figures/5_0.png" in out


def test_format_sources_empty():
    assert _format_sources({"text": [], "images": []}) == "No results."
