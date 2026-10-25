
from fastapi.testclient import TestClient
from src.app import main

client = TestClient(main.app)


def test_query_returns_answer_and_matching_citations(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_chunks",
        lambda: [
            {
                "filename": "machine_learning.txt",
                "page": 1,
                "chunk_id": 1,
                "text": "Supervised learning uses labeled data.",
            }
        ],
    )

    monkeypatch.setattr(
        main,
        "search_chunks",
        lambda question, chunks, top_k: [
            {
                **chunks[0],
                "score": 0.9,
            }
        ],
    )

    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda question, chunks: "Supervised learning uses labeled data. [1]",
    )

    response = client.post(
        "/documents/query",
        json={
            "question": "What is supervised learning?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == "Supervised learning uses labeled data. [1]"
    assert data["result_count"] == 1
    assert data["results"][0]["citation_id"] == 1
    assert data["citations"][0]["citation_id"] == 1
    assert data["citations"][0]["source"] == "machine_learning.txt"
    assert data["citations"][0]["page"] == 1

def test_query_returns_citation_verification_metadata(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_chunks",
        lambda: [
            {
                "filename": "machine_learning.txt",
                "page": 1,
                "chunk_id": 1,
                "text": "Supervised learning uses labeled data.",
            }
        ],
    )

    monkeypatch.setattr(
        main,
        "search_chunks",
        lambda question, chunks, top_k: [
            {**chunks[0], "score": 0.9}
        ],
    )

    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda question, chunks: (
            "Supervised learning uses labeled data. [1]"
        ),
    )

    response = client.post(
        "/documents/query",
        json={
            "question": "What is supervised learning?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert "verification" in data
    assert data["verification"][0]["citation_id"] == 1
    assert data["verification"][0]["status"] == "possible_support"