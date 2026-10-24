
def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[dict]:
    """Split extracted document pages into overlapping text chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    chunks = []
    chunk_id = 1
    step = chunk_size - overlap

    for page in pages:
        text = page["text"].strip()
        page_number = page["page"]

        if not text:
            continue

        for start in range(0, len(text), step):
            chunk_text = text[start : start + chunk_size].strip()

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "text": chunk_text,
                    }
                )
                chunk_id += 1

            if start + chunk_size >= len(text):
                break

    return chunks
