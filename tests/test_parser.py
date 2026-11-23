"""
Tests for Document Parser Service
"""

import io
import pytest
from reportlab.pdfgen import canvas
import docx

from services.parser import (
    parse_document,
    clean_text,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    is_allowed_file,
    sanitize_filename,
    EmptyDocumentError,
    UnreadableDocumentError,
    UnsupportedFormatError,
    DocumentParsingError
)


def create_sample_pdf(text: str) -> bytes:
    """Helper to generate an in-memory PDF with text."""
    stream = io.BytesIO()
    c = canvas.Canvas(stream)
    c.drawString(72, 700, text)
    c.save()
    return stream.getvalue()


def create_sample_docx(text: str) -> bytes:
    """Helper to generate an in-memory DOCX with text."""
    doc = docx.Document()
    doc.add_paragraph(text)
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()


class TestFileValidation:
    def test_allowed_extensions(self):
        assert is_allowed_file("resume.pdf") is True
        assert is_allowed_file("cv.docx") is True
        assert is_allowed_file("profile.txt") is True
        assert is_allowed_file("image.png") is False
        assert is_allowed_file("script.py") is False
        assert is_allowed_file("") is False
        assert is_allowed_file("no_extension") is False

    def test_sanitize_filename(self):
        assert sanitize_filename("../../etc/passwd.pdf") == "etc_passwd.pdf"
        assert sanitize_filename("my resume (1).pdf") == "my_resume_1.pdf"
        assert sanitize_filename("") == "document.txt"


class TestDocumentParsing:
    def test_parse_valid_txt(self):
        sample_text = "Experienced Python and Flask backend developer with SQL skills."
        result = parse_document(sample_text.encode("utf-8"), "resume.txt")
        assert "Python" in result
        assert "Flask" in result

    def test_parse_valid_pdf(self):
        pdf_bytes = create_sample_pdf("Senior DevOps Engineer with Docker and Kubernetes")
        result = parse_document(pdf_bytes, "candidate.pdf")
        assert "DevOps" in result
        assert "Kubernetes" in result

    def test_parse_valid_docx(self):
        docx_bytes = create_sample_docx("Full-Stack React and Node.js Developer")
        result = parse_document(docx_bytes, "applicant.docx")
        assert "React" in result
        assert "Node.js" in result

    def test_empty_file_error(self):
        with pytest.raises(EmptyDocumentError):
            parse_document(b"", "empty.txt")

    def test_unsupported_format_error(self):
        with pytest.raises(UnsupportedFormatError):
            parse_document(b"some content", "data.csv")

    def test_scanned_pdf_error(self):
        # Create PDF canvas with no text
        stream = io.BytesIO()
        c = canvas.Canvas(stream)
        c.showPage()
        c.save()
        with pytest.raises(UnreadableDocumentError):
            parse_document(stream.getvalue(), "scanned.pdf")

    def test_corrupt_pdf_error(self):
        with pytest.raises(DocumentParsingError):
            parse_document(b"%PDF-1.4 corrupt byte stream invalid", "corrupt.pdf")


class TestTextCleaning:
    def test_clean_text_preserves_special_tech_tokens(self):
        raw = "Worked with C++, C#, .NET, and Node.js for 5 years."
        cleaned = clean_text(raw, preserve_symbols=True)
        assert "c++" in cleaned
        assert "c#" in cleaned
        assert ".net" in cleaned
        assert "node.js" in cleaned

    def test_clean_text_normalizes_whitespace(self):
        raw = "Multiple    spaces \n\n and   tabs"
        cleaned = clean_text(raw)
        assert cleaned == "multiple spaces and tabs"
