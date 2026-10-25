import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generate an answer grounded in retrieved document chunks."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not chunks:
        return (
            "I couldn't find relevant information in the uploaded documents."
        )

    # Prepare retrieved passages with numeric citation references.
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[{index}] Source: {chunk.get('filename', 'unknown')}, "
            f"page {chunk.get('page', 'unknown')}\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    # Read the Groq API key from the environment.
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your local .env file."
        )

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    client = Groq(api_key=api_key)

    # Generate an answer grounded in the retrieved passages.
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. "
                    "Answer using only information explicitly supported by "
                    "the supplied context. "
                    "Do not add outside knowledge, assumptions, or "
                    "unsupported explanations. "
                    "If the context does not contain enough evidence to "
                    "answer, say so clearly. "
                    "Attach a citation to each factual claim, using the "
                    "bracketed number of the passage that directly supports "
                    "that claim, such as [1] or [2]. "
                    "Use only the numeric citation format [1], [2], etc. "
                    "Do not use line-based citations or other citation "
                    "formats. "
                    "Do not cite a passage merely because it discusses a "
                    "similar topic. "
                    "Never invent citations, facts, comparisons, or causal "
                    "claims. "
                    "Keep answers concise and close to the wording and "
                    "meaning of the supplied context. "
                    "If only part of a question can be answered from the "
                    "context, answer that part and clearly state what the "
                    "context does not establish."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Retrieved context:\n{context}"
                ),
            },
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("The language model returned an empty answer.")

    answer = answer.strip()

    # Convert line-style citations such as 【1†L3-L5】 to [1].
    answer = re.sub(
        r"【\s*(\d+)(?:†L\d+(?:-L?\d+)?)?\s*】",
        r"[\1]",
        answer,
    )

    max_citation = len(chunks)

    def validate_citation(match: re.Match) -> str:
        """Keep citations that reference an existing retrieved passage."""
        citation_number = int(match.group(1))

        if 1 <= citation_number <= max_citation:
            return match.group(0)

        # Remove citations that reference nonexistent passages.
        return ""

    # Remove numeric citations outside the retrieved chunk range.
    answer = re.sub(
        r"\[(\d+)\]",
        validate_citation,
        answer,
    )

    return answer