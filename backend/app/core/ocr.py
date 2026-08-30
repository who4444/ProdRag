"""OCR for figure images via Tesseract (pytesseract).

Gracefully degrades to "" when the tesseract binary isn't installed (e.g.
local dev before `apt install tesseract-ocr`) so ingestion never hard-fails.
"""

import io
from functools import lru_cache

import pytesseract
from PIL import Image


@lru_cache(maxsize=1)
def _available() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_text(image_bytes: bytes) -> str:
    """Return whitespace-normalized OCR text of the image, or '' if unavailable."""
    if not _available():
        return ""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return " ".join(pytesseract.image_to_string(img).split())
    except Exception:
        return ""