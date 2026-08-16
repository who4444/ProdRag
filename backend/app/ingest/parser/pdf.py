import re
from dataclasses import dataclass, field

import fitz

MIN_FIGURE_AREA = 0.04

# A block counts as a figure/table caption if it starts like "Figure 3:" or
# "Table 2" — real captions beat any bare nearest-block heuristic. Verified
# against ~100 figure/table captions in data/*.pdf.
CAPTION_RE = re.compile(r"^\s*(Figure|Fig\.?|Table|Tab\.?)\s*\d")

# Fallback ceiling (pt) when no caption-like block exists nearby. ponytail:
# heuristic; real captions sit within ~a line-height while paragraph text
# usually doesn't.
CAPTION_MAX_GAP = 15.0


@dataclass
class Block:
    bbox: tuple[float, float, float, float]
    order: int = 0


@dataclass
class TextBlock(Block):
    text: str = ""


@dataclass
class Table(Block):
    index: int = 0
    markdown: str = ""
    rows: list[list[str]] = field(default_factory=list)
    caption: TextBlock | None = None


@dataclass
class Figure(Block):
    index: int = 0
    data: bytes = b""
    width: int = 0
    height: int = 0
    caption: TextBlock | None = None


@dataclass
class Page:
    number: int
    page_image: bytes
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if isinstance(b, TextBlock))

    @property
    def figures(self) -> list[Figure]:
        return [b for b in self.blocks if isinstance(b, Figure)]

    @property
    def tables(self) -> list[Table]:
        return [b for b in self.blocks if isinstance(b, Table)]


@dataclass
class Document:
    pages: list[Page] = field(default_factory=list)


def parse_pdf(data: bytes, dpi: int = 150, min_figure_area: float = MIN_FIGURE_AREA) -> Document:
    """Parse a PDF into a layout tree: Document -> Page -> [TextBlock | Table | Figure].

    Blocks are merged in reading order per page; table cells are not duplicated
    as standalone text, and adjacent text blocks become figure/table captions."""
    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[Page] = []
    try:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            pages.append(
                Page(number=i, page_image=pix.tobytes("png"), blocks=_page_blocks(doc, page, min_figure_area))
            )
    finally:
        doc.close()
    return Document(pages=pages)


def _page_blocks(doc: fitz.Document, page, min_figure_area: float) -> list[Block]:
    text_blocks = [
        (b[0], b[1], b[2], b[3], " ".join(b[4].split()))
        for b in page.get_text("blocks", sort=True)
        if b[6] == 0 and b[4].strip()
    ]
    tables = _extract_tables(page)
    figures = _extract_figures(doc, page, min_figure_area)

    table_boxes = [t.bbox for t in tables]
    kept = [b for b in text_blocks if not any(_inside(b, tb) for tb in table_boxes)]

    remaining = list(kept)
    for obj in sorted(figures + tables, key=lambda o: o.bbox[1]):
        near = _nearest_text_block(obj.bbox, remaining)
        if near is not None:
            obj.caption = TextBlock(bbox=near[:4], text=near[4])
            remaining.remove(near)

    text_blocks = [TextBlock(bbox=b[:4], text=b[4]) for b in remaining]
    blocks: list[Block] = text_blocks + figures + tables
    blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    for order, b in enumerate(blocks):
        b.order = order
    return blocks


def _inside(block: tuple, table_bbox: tuple) -> bool:
    bx0, by0, bx1, by1 = block[:4]
    tx0, ty0, tx1, ty1 = table_bbox
    tol = 2.0
    return bx0 >= tx0 - tol and bx1 <= tx1 + tol and by0 >= ty0 - tol and by1 <= ty1 + tol


def _nearest_text_block(bbox: tuple, candidates: list[tuple]):
    """Nearest horizontally-overlapping block, preferring real captions.

    A "Figure N"/"Table N" block wins regardless of gap (sub-captions like
    "(a)…(b)…" may be closer but aren't captions); otherwise the nearest block
    within CAPTION_MAX_GAP is used as a fallback."""
    x0, y0, x1, y1 = bbox
    obj_h = y1 - y0
    for pick, max_gap in (
        # Captions win, bounded by ~2x object height so a "Figure N" text two
        # objects away (e.g. a table below the figure) can't be stolen.
        (lambda c: CAPTION_RE.match(c[4]), max(CAPTION_MAX_GAP, 2 * obj_h)),
        (lambda c: not CAPTION_RE.match(c[4]), CAPTION_MAX_GAP),
    ):
        best, best_gap = None, float("inf")
        for c in candidates:
            bx0, by0, bx1, by1 = c[:4]
            if bx1 < x0 or bx0 > x1 or not pick(c):
                continue
            gap = by0 - y1 if by0 >= y1 else (y0 - by1 if by1 <= y0 else float("inf"))
            if gap < best_gap and gap <= max_gap:
                best_gap, best = gap, c
        if best is not None:
            return best
    return None


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
            rects = page.get_image_rects(xref)
            bbox = tuple(rects[0]) if rects else (0.0, 0.0, float(w), float(h))
            figures.append(
                Figure(
                    index=len(figures),
                    bbox=bbox,
                    data=info["image"],
                    width=w,
                    height=h,
                )
            )
    return figures


def _extract_tables(page) -> list[Table]:
    tables: list[Table] = []
    text_blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    for i, t in enumerate(page.find_tables().tables):
        markdown = t.to_markdown().strip()
        if not markdown:
            continue
        if _contains_caption(t.bbox, text_blocks):
            # A real table's caption sits above/below its bbox, never inside.
            # If a "Figure/Table N" block is fully inside, this is a figure grid
            # misdetected as a table — drop it so the caption survives.
            continue
        tables.append(
            Table(
                index=i,
                markdown=markdown,
                rows=[list(r) for r in t.extract()],
                bbox=tuple(t.bbox),
            )
        )
    return tables


def _contains_caption(bbox: tuple, text_blocks: list[tuple]) -> bool:
    x0, y0, x1, y1 = bbox
    tol = 1.0
    return any(
        CAPTION_RE.match(b[4])
        and b[0] >= x0 - tol and b[1] >= y0 - tol
        and b[2] <= x1 + tol and b[3] <= y1 + tol
        for b in text_blocks
    )
