from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    size = size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return [c for c in splitter.split_text(text) if c.strip()]
