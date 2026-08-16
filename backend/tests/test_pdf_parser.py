import fitz

from app.ingest.parser.pdf import Document, Figure, TextBlock, parse_pdf


def _make_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 72), "Hello multimodal world")
    page.insert_text((72, 390), "Figure 1: a gray block")
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 300))
    pix.clear_with(200)
    page.insert_image(fitz.Rect(50, 400, 350, 700), pixmap=pix)

    x0, y0, w, h = 72, 100, 60, 20
    for col in range(3):
        for row in range(3):
            page.draw_rect(
                fitz.Rect(x0 + col * w, y0 + row * h, x0 + (col + 1) * w, y0 + (row + 1) * h),
                color=(0, 0, 0),
                width=1,
            )
    cells = [["Q1", "10", "A"], ["Q2", "20", "B"], ["Q3", "30", "C"]]
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            page.insert_text((x0 + c * w + 5, y0 + r * h + 14), val)

    data = doc.tobytes()
    doc.close()
    return data


def test_parse_pdf_returns_document_with_pages():
    doc = parse_pdf(_make_pdf())
    assert isinstance(doc, Document)
    assert len(doc.pages) == 1


def test_parse_pdf_extracts_text_and_figures():
    page = parse_pdf(_make_pdf()).pages[0]
    assert "multimodal" in page.text
    assert page.page_image.startswith(b"\x89PNG")
    assert len(page.figures) >= 1
    assert page.figures[0].data


def test_figure_keeps_caption_context():
    page = parse_pdf(_make_pdf()).pages[0]
    fig = page.figures[0]
    assert isinstance(fig, Figure)
    assert fig.bbox is not None
    assert fig.caption is not None
    assert "Figure 1" in fig.caption.text


def test_parse_pdf_extracts_tables_with_context():
    page = parse_pdf(_make_pdf()).pages[0]
    assert len(page.tables) == 1
    table = page.tables[0]
    assert "|Q1|10|A|" in table.markdown
    assert table.bbox is not None
    assert table.rows and table.rows[0] == ["Q1", "10", "A"]


def test_layout_tree_blocks_in_reading_order():
    page = parse_pdf(_make_pdf()).pages[0]
    blocks = page.blocks
    assert len(blocks) == 3  # intro text, table, figure
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].order == 0
    assert blocks[1].order == 1
    assert blocks[2].order == 2
    assert [b.order for b in blocks] == sorted(b.order for b in blocks)


def test_table_cells_not_duplicated_as_text():
    page = parse_pdf(_make_pdf()).pages[0]
    assert "Q1" not in page.text
    assert "Hello" in page.text
