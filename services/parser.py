"""
Document Parser Service
Handles in-memory extraction of text from PDF, DOCX, and TXT files with
robust error handling for empty, malformed, or scanned documents.
"""

import io
import re
from typing import Optional, Union
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

# Try importing pypdf first, fallback to PyPDF2
try:
    import pypdf as pdf_lib
except ImportError:
    try:
        import PyPDF2 as pdf_lib
    except ImportError:
        pdf_lib = None

# Try importing python-docx, fallback to docx2txt
try:
    import docx
except ImportError:
    docx = None

try:
    import docx2txt
except ImportError:
    docx2txt = None


class DocumentParsingError(Exception):
    """Base exception for document parsing failures."""
    pass


class EmptyDocumentError(DocumentParsingError):
    """Raised when an uploaded document contains no text."""
    pass


class UnreadableDocumentError(DocumentParsingError):
    """Raised when a document appears to be a scanned image with no extractable text."""
    pass


class UnsupportedFormatError(DocumentParsingError):
    """Raised when the document format is not supported."""
    pass


ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


def is_allowed_file(filename: str) -> bool:
    """Check if the filename has an allowed extension."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def sanitize_filename(filename: str) -> str:
    """Return a sanitized, safe filename."""
    clean_name = secure_filename(filename)
    if not clean_name:
        clean_name = "document.txt"
    return clean_name


def extract_text_from_pdf(stream: io.BytesIO) -> str:
    """Extract text from a PDF stream using pypdf or PyPDF2."""
    if pdf_lib is None:
        raise DocumentParsingError("No PDF extraction library available (pypdf/PyPDF2).")

    text_parts = []
    try:
        reader = pdf_lib.PdfReader(stream)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise DocumentParsingError("PDF is password protected and cannot be read.")

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    except Exception as e:
        if isinstance(e, DocumentParsingError):
            raise
        raise DocumentParsingError(f"Corrupt or unreadable PDF: {str(e)}")

    extracted = "\n".join(text_parts).strip()
    if not extracted:
        raise UnreadableDocumentError(
            "Scanned document detected: No extractable text was found in this PDF. "
            "Please provide a text-based PDF or DOCX file."
        )
    return extracted


def extract_text_from_docx(stream: io.BytesIO) -> str:
    """Extract text from a DOCX stream using python-docx with docx2txt fallback."""
    text_parts = []

    # First attempt: python-docx
    if docx is not None:
        try:
            doc = docx.Document(stream)
            # Paragraphs
            for p in doc.paragraphs:
                if p.text.strip():
                    text_parts.append(p.text.strip())
            # Tables
            for table in doc.tables:
                for row in table.rows:
                    row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_texts:
                        text_parts.append(" | ".join(row_texts))
            if text_parts:
                return "\n".join(text_parts).strip()
        except Exception:
            pass

    # Fallback attempt: docx2txt
    if docx2txt is not None:
        try:
            stream.seek(0)
            text = docx2txt.process(stream)
            if text and text.strip():
                return text.strip()
        except Exception:
            pass

    if not text_parts:
        raise EmptyDocumentError("The DOCX document contains no readable text.")
    return "\n".join(text_parts).strip()


def extract_text_from_txt(stream: io.BytesIO) -> str:
    """Extract text from a TXT stream with charset fallbacks."""
    raw_bytes = stream.read()
    if not raw_bytes or not raw_bytes.strip():
        raise EmptyDocumentError("The text document is empty.")

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252", "ascii"):
        try:
            text = raw_bytes.decode(encoding).strip()
            if text:
                return text
        except UnicodeDecodeError:
            continue

    text = raw_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        raise EmptyDocumentError("Could not decode text from file.")
    return text


def parse_document(file_input: Union[FileStorage, bytes, io.BytesIO], filename: str) -> str:
    """
    Parse an in-memory document file (PDF, DOCX, TXT) and return extracted text.
    Handles FileStorage objects, raw bytes, or BytesIO buffers.
    """
    if not filename:
        raise UnsupportedFormatError("No filename provided.")

    clean_name = sanitize_filename(filename)
    if not is_allowed_file(clean_name):
        raise UnsupportedFormatError(f"Unsupported file format for '{clean_name}'. Only PDF, DOCX, and TXT are supported.")

    ext = clean_name.rsplit(".", 1)[1].lower()

    # Read bytes into BytesIO
    if isinstance(file_input, FileStorage):
        content = file_input.read()
    elif isinstance(file_input, bytes):
        content = file_input
    elif isinstance(file_input, io.BytesIO):
        content = file_input.getvalue()
    else:
        raise DocumentParsingError("Invalid file input type.")

    if not content or len(content.strip()) == 0:
        raise EmptyDocumentError(f"Uploaded file '{clean_name}' is empty (0 bytes).")

    stream = io.BytesIO(content)

    if ext == "pdf":
        return extract_text_from_pdf(stream)
    elif ext == "docx":
        return extract_text_from_docx(stream)
    elif ext == "txt":
        return extract_text_from_txt(stream)
    else:
        raise UnsupportedFormatError(f"Unsupported extension: .{ext}")


def clean_text(text: str, preserve_symbols: bool = True) -> str:
    """
    NLP text normalization.
    Preserves special programming tokens like C++, C#, .NET, Node.js if preserve_symbols=True.
    """
    if not text:
        return ""

    text = text.lower()

    if preserve_symbols:
        # Protect specific language tokens
        text = re.sub(r"(?<![a-zA-Z0-9])c\+\+(?![a-zA-Z0-9])", " cpp_lang ", text)
        text = re.sub(r"(?<![a-zA-Z0-9])c\#(?![a-zA-Z0-9])", " csharp_lang ", text)
        text = re.sub(r"(?<![a-zA-Z0-9])\.net(?![a-zA-Z0-9])", " dotnet_framework ", text)
        text = re.sub(r"(?<![a-zA-Z0-9])node\.js(?![a-zA-Z0-9])", " nodejs ", text)
        text = re.sub(r"(?<![a-zA-Z0-9])vue\.js(?![a-zA-Z0-9])", " vuejs ", text)
        text = re.sub(r"(?<![a-zA-Z0-9])react\.js(?![a-zA-Z0-9])", " reactjs ", text)

        # Remove general punctuation except alphanumeric and protected underscores
        text = re.sub(r"[^\w\s]", " ", text)
        # Restore protected tokens
        text = text.replace("cpp_lang", "c++")
        text = text.replace("csharp_lang", "c#")
        text = text.replace("dotnet_framework", ".net")
        text = text.replace("nodejs", "node.js")
        text = text.replace("vuejs", "vue.js")
        text = text.replace("reactjs", "react.js")
    else:
        text = re.sub(r"\W+", " ", text)

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text
