from app.core.ocr import extract_text


def test_extract_text_degrades_gracefully_when_unavailable():
    # tesseract binary may not be installed in CI/dev; must never raise.
    text = extract_text(b"\x89PNG\r\n\x1a\nnot a real image")
    assert isinstance(text, str)