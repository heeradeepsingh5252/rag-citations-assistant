
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from src.app.chunker import chunk_pages
from src.app.config import APP_NAME
from src.app.document_loader import DocumentIngestionError, extract_document

app = FastAPI(title=APP_NAME)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": APP_NAME}


@app.post("/documents/ingest")
async def ingest_document(file: Annotated[UploadFile, File()]):
    filename = file.filename or ""

    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a PDF or TXT file.",
        )

    content = await file.read()

    # Limit uploads to 10 MB.
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

    return {
        "filename": filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "pages": pages,
        "chunks": chunks,
    }
