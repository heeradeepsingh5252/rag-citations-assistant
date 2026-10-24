
from unittest.mock import patch

import pytest
from src.app.search import search_chunks


def test_empty_chunks_return_empty_results():
    assert search_chunks("What is RAG?", []) == []


def test_blank_query_is_rejected():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        search_chunks("   ", [])


def test_invalid_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k must be positive"):
        search_chunks("What is RAG?", [], top_k=0)


def test_relevant_chunks_are_ranked_first():
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "Cats are domestic animals."},
        {"chunk_id": 2, "page": 2, "text": "RAG retrieves documents to answer questions."},
    ]

    # Mock embeddings so this test does not load the ML model.
    with (
        patch("src.app.search.embed_texts") as mock_texts,
        patch("src.app.search.embed_query") as mock_query,
    ):
        mock_texts.return_value = [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        mock_query.return_value = [1.0, 0.0]

        results = search_chunks("How does RAG work?", chunks, top_k=2)

    assert results[0]["chunk_id"] == 2
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["chunk_id"] == 1
