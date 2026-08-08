from app.chunking import chunk_text


def test_chunk_text_covers_and_overlaps():
    text = "abcdefghij" * 100
    chunks = chunk_text(text, size=40, overlap=10)
    assert chunks[0] == text[:40]
    assert all(c in text for c in chunks)
    assert chunks[-1][-1] == text[-1]
    assert len(chunks) > 1


def test_chunk_text_empty():
    assert chunk_text("   ") == []
