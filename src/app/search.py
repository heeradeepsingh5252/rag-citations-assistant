
from src.app.embeddings import embed_query, embed_texts


def search_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Return the most relevant unique chunks using cosine similarity."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if not chunks:
        return []

    # Remove exact duplicate text while preserving the first occurrence.
    unique_chunks = []
    seen_texts = set()

    for chunk in chunks:
        text = chunk["text"].strip()
        if not text or text in seen_texts:
            continue

        seen_texts.add(text)
        unique_chunks.append(chunk)

    if not unique_chunks:
        return []

    texts = [chunk["text"] for chunk in unique_chunks]
    chunk_embeddings = embed_texts(texts)
    query_embedding = embed_query(query)

    # Normalized embeddings make dot product equal cosine similarity.
    scored_chunks = []

    for chunk, embedding in zip(unique_chunks, chunk_embeddings):
        score = sum(
            query_value * chunk_value
            for query_value, chunk_value in zip(query_embedding, embedding)
        )
        scored_chunks.append({**chunk, "score": float(score)})

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    # Filter chunks that fall below the minimum relevance threshold.
    min_score = 0.20
    relevant_chunks = [
        chunk for chunk in scored_chunks
        if chunk["score"] >= min_score
    ]

    return relevant_chunks[:top_k]
