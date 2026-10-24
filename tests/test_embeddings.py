
import pytest
from src.app.embeddings import embed_query, embed_texts


def test_empty_text_list_returns_empty_embeddings():
    assert embed_texts([]) == []


def test_blank_query_is_rejected():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        embed_query("   ")
