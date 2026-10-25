
from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(MODEL_NAME, local_files_only=True)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert document chunks into numerical embeddings."""
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Convert a user's question into an embedding."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    return embed_texts([query])[0]

