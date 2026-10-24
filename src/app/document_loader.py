
from io import BytesIO

from pypdf import PdfReader


class DocumentIngestionError(ValueError):
    """Raised when a document cannot be ingested."""


def extract_document(filename: str, content: bytes) -> list[dict]:
    """Extract text from a PDF or TXT file, preserving page numbers where possible."""
    if not content:
        raise DocumentIngestionError("The uploaded file is empty.")

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension == "txt":
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise DocumentIngestionError(
                "TXT files must use UTF-8 encoding."
            ) from exc

        if not text:
            raise DocumentIngestionError("No readable text was found.")

        return [{"page": 1, "text": text}]

    if extension == "pdf":
        try:
            reader = PdfReader(BytesIO(content))
            pages = []

            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append({"page": page_number, "text": text})

        except Exception as exc:
            raise DocumentIngestionError(
                "The PDF could not be read. It may be corrupted or encrypted."
            ) from exc

        if not pages:
            raise DocumentIngestionError(
                "No extractable text was found in the PDF. It may be scanned."
            )

        return pages

    raise DocumentIngestionError("Unsupported file type. Upload a PDF or TXT file.")
