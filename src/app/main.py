import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.app.answer_generator import generate_answer
from src.app.chunker import chunk_pages
from src.app.citation_verifier import (
    check_claim_support,
    get_cited_chunks,
)
from src.app.config import APP_NAME
from src.app.document_loader import DocumentIngestionError, extract_document
from src.app.search import search_chunks
from src.app.store import get_chunks, save_chunks

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_DIR / "static"
SAMPLE_DOCUMENT = PROJECT_DIR / "data" / "machine_learning.txt"


def preload_sample_document() -> None:
    """Load the sample document when the in-memory store is empty."""
    if get_chunks():
        logger.info(
            "Document store already contains chunks; skipping preload."
        )
        return

    if not SAMPLE_DOCUMENT.is_file():
        logger.warning(
            "Sample document not found at %s. "
            "Users must upload a document manually.",
            SAMPLE_DOCUMENT,
        )
        return

    try:
        pages = extract_document(
            SAMPLE_DOCUMENT.name,
            SAMPLE_DOCUMENT.read_bytes(),
        )
        chunks = chunk_pages(pages)
    except (OSError, DocumentIngestionError):
        logger.exception("Failed to preload the sample document.")
        return

    if not chunks:
        logger.warning("Sample document produced no chunks.")
        return

    save_chunks(chunks, filename=SAMPLE_DOCUMENT.name)

    logger.info(
        "Preloaded %s: %d page(s), %d chunk(s).",
        SAMPLE_DOCUMENT.name,
        len(pages),
        len(chunks),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize application resources during startup."""
    preload_sample_document()
    yield


app = FastAPI(title=APP_NAME, lifespan=lifespan)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the web interface."""
    return FileResponse(STATIC_DIR / "index.html")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    """Return the application's health status."""
    return {"status": "ok", "app": APP_NAME}


@app.post("/documents/ingest")
async def ingest_document(file: Annotated[UploadFile, File()]) -> dict:
    """Extract an uploaded PDF or TXT document and store its chunks."""
    filename = file.filename or ""

    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a PDF or TXT file.",
        )

    try:
        content = await file.read()

        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 10 MB limit.",
            )

        try:
            pages = extract_document(filename, content)
        except DocumentIngestionError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
    finally:
        await file.close()

    chunks = chunk_pages(pages)
    save_chunks(chunks, filename=filename)

    return {
        "filename": filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "pages": pages,
        "chunks": chunks,
    }


@app.post("/documents/query")
def query_documents(request: QueryRequest) -> dict:
    """Retrieve relevant chunks and generate a cited answer."""
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    chunks = get_chunks()

    if not chunks:
        return {
            "question": question,
            "answer": None,
            "result_count": 0,
            "results": [],
            "citations": [],
            "verification": [],
            "message": (
                "No documents indexed yet. Upload a PDF or TXT file first."
            ),
        }

    # Generate embeddings and retrieve relevant document passages.
    try:
        results = search_chunks(
            question,
            chunks,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.exception("Embedding or retrieval failed.")
        raise HTTPException(
            status_code=502,
            detail=(
                "Embedding or retrieval failed. Check Gemini API access, "
                "model availability, and network connectivity."
            ),
        ) from exc

    cited_results = []
    citations = []

    # Assign citation numbers to the retrieved passages.
    for index, result in enumerate(results, start=1):
        citation = {
            "citation_id": index,
            "source": result["filename"],
            "page": result["page"],
            "chunk_id": result["chunk_id"],
        }

        cited_results.append(
            {
                **result,
                "citation_id": index,
            }
        )
        citations.append(citation)

    # Avoid calling the language model when nothing relevant was found.
    if not cited_results:
        return {
            "question": question,
            "answer": (
                "I couldn't find relevant information in the uploaded "
                "documents."
            ),
            "result_count": 0,
            "results": [],
            "citations": [],
            "verification": [],
        }

    # Generate an answer using the retrieved passages.
    try:
        answer = generate_answer(question, cited_results)
    except RuntimeError as exc:
        logger.exception("Answer generation failed.")
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Answer generation failed.")
        raise HTTPException(
            status_code=502,
            detail=(
                "Answer generation failed. Check Groq API access, "
                "model availability, and your connection."
            ),
        ) from exc

    # Verify each source passage explicitly cited by the answer.
    cited_chunks = get_cited_chunks(answer, cited_results)
    verification = []

    for item in cited_chunks:
        citation_number = item["citation_number"]
        chunk = item["chunk"]

        verification.append(
            {
                "citation_id": citation_number,
                "source": chunk["filename"],
                "page": chunk["page"],
                "status": check_claim_support(
                    answer,
                    chunk["text"],
                ),
                "method": "lexical_heuristic",
            }
        )

    return {
        "question": question,
        "answer": answer,
        "result_count": len(cited_results),
        "results": cited_results,
        "citations": citations,
        "verification": verification,
    }