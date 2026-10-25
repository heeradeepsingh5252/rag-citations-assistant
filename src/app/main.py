
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from src.app.answer_generator import generate_answer
from src.app.chunker import chunk_pages
from src.app.config import APP_NAME
from src.app.document_loader import DocumentIngestionError, extract_document
from src.app.search import search_chunks
from src.app.store import get_chunks, save_chunks

app = FastAPI(title=APP_NAME)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": APP_NAME}


@app.post("/documents/ingest")
async def ingest_document(file: Annotated[UploadFile, File()]) -> dict:
    filename = file.filename or ""

    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a PDF or TXT file.",
        )

    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MB limit.",
        )

    try:
        pages = extract_document(filename, content)
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            "message": "No documents indexed yet. Upload a PDF or TXT file first.",
        }

    results = search_chunks(question, chunks, top_k=request.top_k)

    cited_results = []
    citations = []

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

    try:
        answer = generate_answer(question, cited_results)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Answer generation failed. Check Groq API access, "
                "model availability, and your connection."
            ),
        ) from exc

    return {
        "question": question,
        "answer": answer,
        "result_count": len(cited_results),
        "results": cited_results,
        "citations": citations,
    }
