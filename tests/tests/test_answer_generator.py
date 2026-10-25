from types import SimpleNamespace

import pytest
from src.app import answer_generator
from src.app.answer_generator import generate_answer


def test_empty_chunks_return_fallback_message():
    answer = generate_answer("What is machine learning?", [])
    assert "couldn't find relevant information" in answer


def test_blank_question_raises_error():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        generate_answer("   ", [])


def test_missing_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    chunks = [
        {
            "filename": "machine_learning.txt",
            "page": 1,
            "chunk_id": 1,
            "text": "Supervised learning uses labeled data.",
        }
    ]

    with pytest.raises(RuntimeError, match="GROQ_API_KEY is missing"):
        generate_answer("What is supervised learning?", chunks)


def test_generate_answer_uses_mocked_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="Supervised learning uses labeled data. [1]"
                        )
                    )
                ]
            )

    class FakeGroq:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(answer_generator, "Groq", FakeGroq)

    chunks = [
        {
            "filename": "machine_learning.txt",
            "page": 1,
            "chunk_id": 1,
            "text": "Supervised learning uses labeled data.",
        }
    ]

    answer = answer_generator.generate_answer(
        "What is supervised learning?",
        chunks,
    )

    assert answer == "Supervised learning uses labeled data. [1]"
    assert captured["model"] == "test-model"
    assert "Supervised learning uses labeled data." in (captured["messages"][1]["content"])


def test_invalid_citation_numbers_are_removed(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="Supervised learning uses labeled data. [1] [7]"
                        )
                    )
                ]
            )

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(answer_generator, "Groq", FakeGroq)

    chunks = [
        {
            "filename": "machine_learning.txt",
            "page": 1,
            "chunk_id": 1,
            "text": "Supervised learning uses labeled data.",
        }
    ]

    answer = generate_answer(
        "What is supervised learning?",
        chunks,
    )

    assert "[1]" in answer
    assert "[7]" not in answer


def test_multiple_valid_and_invalid_citations(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-model")

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="Supervised learning uses labeled data [1]. "
                            "It is always faster than unsupervised learning [9]."
                        )
                    )
                ]
            )

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(answer_generator, "Groq", FakeGroq)

    chunks = [
        {
            "filename": "machine_learning.txt",
            "page": 1,
            "chunk_id": 1,
            "text": "Supervised learning uses labeled data.",
        }
    ]

    answer = answer_generator.generate_answer(
        "What is supervised learning?",
        chunks,
    )

    assert "[1]" in answer
    assert "[9]" not in answer
