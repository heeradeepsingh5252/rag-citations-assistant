import json
from pathlib import Path

from src.app.chunker import chunk_pages
from src.app.document_loader import extract_document
from src.app.search import search_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "machine_learning.txt"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"


def main() -> None:
    pages = extract_document(
        DATA_PATH.name,
        DATA_PATH.read_bytes(),
    )
    chunks = chunk_pages(pages)

    questions = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    evidence_checks = []
    out_of_scope_checks = []

    for item in questions:
        results = search_chunks(item["question"], chunks, top_k=3)
        retrieved_text = " ".join(
            result["text"].lower() for result in results
        )

        if item["topic"] == "out_of_scope":
            passed = len(results) == 0
            out_of_scope_checks.append(passed)
        else:
            expected_evidence = item["expected_evidence"].lower()
            passed = expected_evidence in retrieved_text
            evidence_checks.append(passed)

        status = "PASS" if passed else "FAIL"
        print(f'{status} {item["id"]}: {item["question"]}')
        print(f"  Retrieved chunks: {len(results)}")

        if results:
            print(f'  Top score: {results[0]["score"]:.4f}')

    print("\n=== Evaluation Summary ===")

    evidence_total = len(evidence_checks)
    evidence_passed = sum(evidence_checks)
    evidence_rate = (
        evidence_passed / evidence_total if evidence_total else 0.0
    )

    rejection_total = len(out_of_scope_checks)
    rejection_passed = sum(out_of_scope_checks)
    rejection_rate = (
        rejection_passed / rejection_total if rejection_total else 0.0
    )

    print(
        f"Evidence retrieval: {evidence_passed}/{evidence_total} "
        f"({evidence_rate:.1%})"
    )
    print(
        f"Out-of-scope rejection: {rejection_passed}/{rejection_total} "
        f"({rejection_rate:.1%})"
    )


if __name__ == "__main__":
    main()
