
from src.app.embeddings import embed_query, embed_texts


def search_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Return the most relevant chunks for a query using cosine similarity."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_texts(texts)
    query_embedding = embed_query(query)

    # Embeddings are normalized, so dot product equals cosine similarity.
    scored_chunks = []

    for chunk, embedding in zip(chunks, chunk_embeddings):
        score = sum(
            query_value * chunk_value
            for query_value, chunk_value in zip(query_embedding, embedding)
        )
        scored_chunks.append({**chunk, "score": float(score)})

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    return scored_chunks[:top_k]
