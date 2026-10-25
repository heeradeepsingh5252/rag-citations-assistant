
import pytest
from src.app.citation_verifier import (
    check_claim_support,
    extract_citation_numbers,
    get_cited_chunks,
)


def test_extract_citation_numbers():
    assert extract_citation_numbers(
        "Fact A [1]. Fact B [2]. Fact C [1]."
    ) == [1, 2]


def test_extract_citation_numbers_without_citations():
    assert extract_citation_numbers("No citations here.") == []


def test_get_cited_chunks_ignores_invalid_citations():
    chunks = [
        {"text": "First source", "page": 1},
        {"text": "Second source", "page": 2},
    ]

    result = get_cited_chunks(
        "Claim [2]. Unsupported reference [9].", chunks
    )

    assert len(result) == 1
    assert result[0]["citation_number"] == 2
    assert result[0]["chunk"]["text"] == "Second source"


def test_get_cited_chunks_with_no_citations():
    chunks = [{"text": "First source", "page": 1}]

    assert get_cited_chunks("No citation.", chunks) == []


def test_claim_with_matching_words_is_possible_support():
    claim = "Supervised learning uses labeled training data"
    source = (
        "Supervised learning trains models using labeled "
        "training data."
    )

    assert check_claim_support(claim, source) == "possible_support"


def test_unrelated_claim_has_insufficient_evidence():
    claim = "The capital of Japan is Tokyo"
    source = (
        "Supervised learning uses labeled data to train "
        "machine learning models."
    )

    assert check_claim_support(claim, source) == "insufficient_evidence"


def test_empty_claim_raises_error():
    with pytest.raises(ValueError):
        check_claim_support("", "Some source text")


def test_empty_source_raises_error():
    with pytest.raises(ValueError):
        check_claim_support("Some claim", "")


def test_contradictory_source_is_not_confirmed_as_support():
    claim = "Supervised learning uses labeled training data"
    source = (
        "Supervised learning does not use labeled "
        "training data."
    )

    assert check_claim_support(
        claim, source
    ) == "insufficient_evidence"
