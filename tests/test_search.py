from unittest.mock import patch

import pytest
from src.app.search import search_chunks


def test_relevant_chunks_are_ranked_first():
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "Cats are domestic animals."},
        {"chunk_id": 2, "page": 2, "text": "RAG retrieves documents to answer questions."},
    ]

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

    assert len(results) == 1
    assert results[0]["chunk_id"] == 2
    assert results[0]["score"] == pytest.approx(1.0)


def test_duplicate_text_is_removed_before_embedding():
    chunks = [
        {"chunk_id": 1, "page": 1, "text": "RAG retrieves relevant documents."},
        {"chunk_id": 2, "page": 2, "text": "RAG retrieves relevant documents."},
        {"chunk_id": 3, "page": 3, "text": "Language models generate answers."},
    ]

    with (
        patch("src.app.search.embed_texts") as mock_texts,
        patch("src.app.search.embed_query") as mock_query,
    ):
        mock_texts.return_value = [
            [1.0, 0.0],
            [0.8, 0.6],
        ]
        mock_query.return_value = [1.0, 0.0]

        results = search_chunks("How does RAG work?", chunks, top_k=3)

    assert len(results) == 2
    assert [result["chunk_id"] for result in results] == [1, 3]
    assert mock_texts.call_args.args[0] == [
        "RAG retrieves relevant documents.",
        "Language models generate answers.",
    ]
