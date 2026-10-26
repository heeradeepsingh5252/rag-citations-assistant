
import math
import os
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 768


@lru_cache(maxsize=1)
def get_embedding_client() -> genai.Client:
    """Create and reuse the Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your local .env file."
        )

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _normalize_embeddings(
    embeddings: list[list[float]],
) -> list[list[float]]:
    """Normalize vectors so dot product represents cosine similarity."""
    normalized = []

    for embedding in embeddings:
        vector = [float(value) for value in embedding]
        norm = math.sqrt(sum(value * value for value in vector))

        if norm == 0:
            raise RuntimeError("The embedding model returned a zero vector.")

        normalized.append([value / norm for value in vector])

    return normalized


def _embed(
    texts: list[str],
    task_type: str,
) -> list[list[float]]:
    """Generate normalized embeddings using the Gemini API."""
    if not texts:
        return []

    client = get_embedding_client()

    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=OUTPUT_DIMENSIONALITY,
        ),
    )

    raw_embeddings = [
        item.values
        for item in (response.embeddings or [])
        if item.values is not None
    ]

    if len(raw_embeddings) != len(texts):
        raise RuntimeError(
            "Gemini returned an unexpected number of embeddings."
        )

    return _normalize_embeddings(raw_embeddings)


@lru_cache(maxsize=16)
def _cached_document_embeddings(
    texts: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    """Cache document embeddings for identical batches of text."""
    vectors = _embed(list(texts), "RETRIEVAL_DOCUMENT")

    # Store immutable tuples inside the cache.
    return tuple(tuple(vector) for vector in vectors)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate and cache embeddings for document chunks."""
    if not texts:
        return []

    cached_vectors = _cached_document_embeddings(tuple(texts))

    # Return fresh lists so callers cannot mutate cached vectors.
    return [list(vector) for vector in cached_vectors]


def embed_query(query: str) -> list[float]:
    """Generate an embedding for a user's search question."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    embeddings = _embed([query], "RETRIEVAL_QUERY")
    return embeddings[0]
