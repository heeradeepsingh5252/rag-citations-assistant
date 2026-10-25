import re


def extract_citation_numbers(text: str) -> list[int]:
    """Extract unique numeric citation references from text."""
    return list(
        dict.fromkeys(
            int(match)
            for match in re.findall(r"\[(\d+)\]", text)
        )
    )


def get_cited_chunks(
    text: str,
    chunks: list[dict],
) -> list[dict]:
    """Return chunks referenced by valid numeric citations."""
    citation_numbers = extract_citation_numbers(text)

    return [
        {
            "citation_number": number,
            "chunk": chunks[number - 1],
        }
        for number in citation_numbers
        if 1 <= number <= len(chunks)
    ]


def check_claim_support(claim: str, source_text: str) -> str:
    """
    Check lexical support across relevant source sentences.

    This is a heuristic, not a semantic fact-checker.
    """
    if not claim.strip() or not source_text.strip():
        raise ValueError("Claim and source text must not be empty.")

    def normalize(text: str) -> list[str]:
        return re.findall(r"\b[\w'-]+\b", text.lower())

    negations = {"not", "no", "never", "neither", "nor", "without"}

    # Remove numeric citation markers before checking the answer.
    claim = re.sub(r"\[\d+\]", "", claim).strip()

    # Split the answer into individual sentences.
    claim_sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", claim)
        if sentence.strip()
    ]

    # Split the source text into individual sentences.
    source_sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", source_text)
        if sentence.strip()
    ]

    for claim_sentence in claim_sentences:
        claim_words = normalize(claim_sentence)

        meaningful_words = {
            word
            for word in claim_words
            if len(word) > 3 and word not in negations
        }

        if not meaningful_words:
            return "insufficient_evidence"

        claim_has_negation = any(
            word in negations for word in claim_words
        )

        # Collect overlapping evidence from compatible source sentences.
        supporting_source_words = set()

        for source_sentence in source_sentences:
            source_words = normalize(source_sentence)

            source_has_negation = any(
                word in negations for word in source_words
            )

            # Avoid matching sentences with different negation status.
            if claim_has_negation != source_has_negation:
                continue

            # Include only sentences that overlap with the claim.
            if meaningful_words & set(source_words):
                supporting_source_words.update(source_words)

        # Measure how much of the claim is supported by word overlap.
        matched_words = meaningful_words & supporting_source_words
        coverage = len(matched_words) / len(meaningful_words)

        if coverage < 0.60:
            return "insufficient_evidence"

    return "possible_support"