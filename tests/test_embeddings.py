import pytest
from src.app import embeddings
from src.app.embeddings import embed_query, embed_texts


def test_empty_text_list_returns_empty_embeddings():
    assert embed_texts([]) == []


def test_blank_query_is_rejected():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        embed_query("   ")


def test_repeated_document_batches_use_cached_embeddings(monkeypatch):
    embeddings._cached_document_embeddings.cache_clear()
    calls = []

    def fake_embed(texts, task_type):
        calls.append((tuple(texts), task_type))
        return [
            [float(index + 1), float(index + 2)]
            for index, _ in enumerate(texts)
        ]

    monkeypatch.setattr(embeddings, "_embed", fake_embed)

    texts = ["sample document one", "sample document two"]

    try:
        first = embed_texts(texts)
        second = embed_texts(texts)

        assert first == second
        assert len(calls) == 1
        assert calls[0][1] == "RETRIEVAL_DOCUMENT"
    finally:
        embeddings._cached_document_embeddings.cache_clear()