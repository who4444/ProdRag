import fitz

from app.parser.pdf import parse_pdf


def _make_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Hello multimodal world")
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 300))
    pix.clear_with(200)
    page.insert_image(fitz.Rect(50, 400, 350, 700), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


def test_parse_pdf_extracts_text_and_figures():
    pages = parse_pdf(_make_pdf())
    assert len(pages) == 1
    assert "multimodal" in pages[0].text
    assert pages[0].page_image.startswith(b"\x89PNG")
    assert len(pages[0].figures) >= 1
    assert pages[0].figures[0].data
