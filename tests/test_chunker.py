
import pytest
from src.app.chunker import chunk_pages


def test_short_text_creates_one_chunk():
    pages = [{"page": 1, "text": "RAG retrieves relevant documents."}]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0]["page"] == 1
    assert chunks[0]["text"] == "RAG retrieves relevant documents."


def test_long_text_creates_overlapping_chunks():
    pages = [{"page": 3, "text": "A" * 120}]

    chunks = chunk_pages(pages, chunk_size=50, overlap=10)

    assert len(chunks) == 3
    assert chunks[0]["text"][-10:] == chunks[1]["text"][:10]
    assert all(chunk["page"] == 3 for chunk in chunks)


def test_invalid_chunk_settings_are_rejected():
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=10, overlap=10)
