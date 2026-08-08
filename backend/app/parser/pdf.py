from dataclasses import dataclass

import fitz

MIN_FIGURE_AREA = 0.04


@dataclass
class Figure:
    index: int
    data: bytes
    width: int
    height: int


@dataclass
class Page:
    number: int
    text: str
    page_image: bytes
    figures: list[Figure]


def parse_pdf(data: bytes, dpi: int = 150, min_figure_area: float = MIN_FIGURE_AREA) -> list[Page]:
    """Parse a PDF into pages: extracted text, a full-page render, and embedded figures."""
    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[Page] = []
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            pages.append(
                Page(
                    number=i,
                    text=page.get_text("text"),
                    page_image=pix.tobytes("png"),
                    figures=_extract_figures(doc, page, min_figure_area),
                )
            )
    finally:
        doc.close()
    return pages


def _extract_figures(doc: fitz.Document, page, min_figure_area: float) -> list[Figure]:
    figures: list[Figure] = []
    page_area = page.rect.width * page.rect.height
    seen: set[tuple] = set()
    for xref, _smask, w, h, *_rest in page.get_images(full=True):
        if not xref or w * h < min_figure_area * page_area:
            continue
        key = (xref, w, h)
        if key in seen:
            continue
        seen.add(key)
        info = doc.extract_image(xref)
        if info:
            figures.append(Figure(len(figures), info["image"], w, h))
    return figures
