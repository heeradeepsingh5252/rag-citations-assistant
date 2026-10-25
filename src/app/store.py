
from threading import Lock

_store: list[dict] = []
_lock = Lock()


def save_chunks(chunks: list[dict], filename: str = "unknown") -> None:
    """Add document chunks to the in-memory store with source metadata."""
    tagged_chunks = [
        {**chunk, "filename": filename}
        for chunk in chunks
    ]

    with _lock:
        _store.extend(tagged_chunks)


def get_chunks() -> list[dict]:
    """Return a snapshot of all stored chunks."""
    with _lock:
        return [chunk.copy() for chunk in _store]


def clear_store() -> None:
    """Clear stored chunks, mainly for testing."""
    with _lock:
        _store.clear()
