
import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generate an answer using only the retrieved document chunks."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not chunks:
        return "I couldn't find relevant information in the uploaded documents."

    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[{index}] Source: {chunk.get('filename', 'unknown')}, "
            f"page {chunk.get('page', 'unknown')}\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your local .env file."
        )

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. "
                    "Answer using only the supplied context. "
                    "If the context does not contain the answer, say so. "
                    "Cite supporting passages using their bracketed numbers, "
                    "such as [1] or [2]. Never invent citations."
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
    max_citation = len(chunks)

    def validate_citation(match: re.Match) -> str:
        citation_number = int(match.group(1))
        if 1 <= citation_number <= max_citation:
            return match.group(0)
        return ""

    answer = re.sub(r"\[(\d+)\]", validate_citation, answer)

    return answer