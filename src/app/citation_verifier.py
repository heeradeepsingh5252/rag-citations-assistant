
import re


def extract_citation_numbers(text: str) -> list[int]:
    """Extract unique numeric citation references from text."""
    return list(
        dict.fromkeys(
            int(match) for match in re.findall(r"\[(\d+)\]", text)
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
    """Perform a basic lexical support check with negation detection."""
    if not claim.strip() or not source_text.strip():
        raise ValueError("Claim and source text must not be empty.")

    def normalize(text: str) -> list[str]:
        return re.findall(r"\b[\w'-]+\b", text.lower())

    claim_words = normalize(claim)
    source_words = normalize(source_text)

    negations = {"not", "no", "never", "neither", "nor", "without"}

    claim_has_negation = any(word in negations for word in claim_words)
    source_has_negation = any(word in negations for word in source_words)

    # Don't treat a simple polarity mismatch as supporting evidence.
    if claim_has_negation != source_has_negation:
        return "insufficient_evidence"

    meaningful_words = {
        word for word in claim_words
        if len(word) > 3 and word not in negations
    }

    if not meaningful_words:
        return "insufficient_evidence"

    matched_words = meaningful_words & set(source_words)
    coverage = len(matched_words) / len(meaningful_words)

    if coverage >= 0.6:
        return "possible_support"

    return "insufficient_evidence"
