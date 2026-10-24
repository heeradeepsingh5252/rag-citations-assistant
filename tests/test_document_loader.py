
from io import BytesIO

import pytest
from pypdf import PdfWriter
from src.app.document_loader import DocumentIngestionError, extract_document


def test_extract_txt():
    result = extract_document("notes.txt", b"RAG retrieves relevant documents.")
    assert result == [{"page": 1, "text": "RAG retrieves relevant documents."}]


def test_reject_empty_file():
    with pytest.raises(DocumentIngestionError, match="empty"):
        extract_document("notes.txt", b"")


def test_reject_unsupported_file():
    with pytest.raises(DocumentIngestionError, match="Unsupported"):
        extract_document("notes.docx", b"some content")


def test_extract_pdf_page():
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    pdf_buffer = BytesIO()
    writer.write(pdf_buffer)

    # A blank PDF has no extractable text, so it should be rejected.
    with pytest.raises(DocumentIngestionError, match="No extractable text"):
        extract_document("blank.pdf", pdf_buffer.getvalue())

