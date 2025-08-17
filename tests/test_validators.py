"""Tests for the validators module.

This module contains comprehensive tests for PDF validation functionality
including file size checks, format validation, and extension verification.
"""

import pytest
from pathlib import Path

from src.pdf_parser.validators import PDFValidator
from src.pdf_parser.exceptions import ValidationError
from src.pdf_parser.config import Config


class TestPDFValidator:
    """Test cases for PDFValidator class."""

    def test_validate_pdf_file_success(self, sample_pdf_bytes):
        """Test successful PDF file validation."""
        # Should not raise any exception
        PDFValidator.validate_pdf_file(sample_pdf_bytes, "test.pdf")

    def test_validate_pdf_file_invalid_format(self, invalid_pdf_bytes):
        """Test validation with invalid PDF format."""
        with pytest.raises(ValidationError, match="is not a valid PDF file"):
            PDFValidator.validate_pdf_file(invalid_pdf_bytes, "test.pdf")

    def test_validate_pdf_file_too_large(self, too_large_pdf_bytes):
        """Test validation with file too large."""
        with pytest.raises(ValidationError, match="is too large"):
            PDFValidator.validate_pdf_file(too_large_pdf_bytes, "large.pdf")

    def test_validate_pdf_file_too_small(self, too_small_pdf_bytes):
        """Test validation with file too small."""
        with pytest.raises(ValidationError, match="is too small or corrupted"):
            PDFValidator.validate_pdf_file(too_small_pdf_bytes, "small.pdf")

    def test_validate_pdf_file_invalid_extension(self, sample_pdf_bytes):
        """Test validation with invalid file extension."""
        with pytest.raises(ValidationError, match="Invalid file extension"):
            PDFValidator.validate_pdf_file(sample_pdf_bytes, "test.txt")

    def test_validate_file_size_success(self, sample_pdf_bytes):
        """Test successful file size validation."""
        # Should not raise any exception
        PDFValidator._validate_file_size(sample_pdf_bytes, "test.pdf")

    def test_validate_file_size_too_large(self):
        """Test file size validation with oversized file."""
        large_content = b"x" * (Config.MAX_FILE_SIZE + 1)
        with pytest.raises(ValidationError) as exc_info:
            PDFValidator._validate_file_size(large_content, "large.pdf")

        assert "is too large" in str(exc_info.value)
        assert f"{Config.MAX_FILE_SIZE // (1024*1024)}MB" in str(exc_info.value)

    def test_validate_file_size_too_small(self):
        """Test file size validation with undersized file."""
        small_content = b"x" * (Config.MIN_FILE_SIZE - 1)
        with pytest.raises(ValidationError) as exc_info:
            PDFValidator._validate_file_size(small_content, "small.pdf")

        assert "is too small or corrupted" in str(exc_info.value)

    def test_validate_file_size_boundary_conditions(self):
        """Test file size validation at boundary conditions."""
        # Test minimum size boundary
        min_content = b"x" * Config.MIN_FILE_SIZE
        PDFValidator._validate_file_size(min_content, "min.pdf")  # Should not raise

        # Test maximum size boundary
        max_content = b"x" * Config.MAX_FILE_SIZE
        PDFValidator._validate_file_size(max_content, "max.pdf")  # Should not raise

    def test_validate_pdf_format_success(self, sample_pdf_bytes):
        """Test successful PDF format validation."""
        # Should not raise any exception
        PDFValidator._validate_pdf_format(sample_pdf_bytes, "test.pdf")

    def test_validate_pdf_format_invalid(self):
        """Test PDF format validation with invalid content."""
        invalid_content = b"This is not a PDF"
        with pytest.raises(ValidationError, match="is not a valid PDF file"):
            PDFValidator._validate_pdf_format(invalid_content, "invalid.pdf")

    def test_validate_pdf_format_empty(self):
        """Test PDF format validation with empty content."""
        with pytest.raises(ValidationError, match="is not a valid PDF file"):
            PDFValidator._validate_pdf_format(b"", "empty.pdf")

    def test_validate_pdf_format_partial_header(self):
        """Test PDF format validation with partial PDF header."""
        partial_content = b"%PD"
        with pytest.raises(ValidationError, match="is not a valid PDF file"):
            PDFValidator._validate_pdf_format(partial_content, "partial.pdf")

    def test_validate_file_extension_success(self):
        """Test successful file extension validation."""
        # Should not raise any exception
        PDFValidator._validate_file_extension("document.pdf")
        PDFValidator._validate_file_extension("Document.PDF")
        PDFValidator._validate_file_extension("file.Pdf")

    def test_validate_file_extension_invalid(self):
        """Test file extension validation with invalid extensions."""
        invalid_extensions = ["file.txt", "document.doc", "image.png", "file", "test.pdf.txt"]

        for filename in invalid_extensions:
            with pytest.raises(ValidationError) as exc_info:
                PDFValidator._validate_file_extension(filename)

            expected_suffix = Path(filename).suffix
            assert "Invalid file extension" in str(exc_info.value)
            assert f"got: {expected_suffix}" in str(exc_info.value)

    def test_validate_file_extension_complex_names(self):
        """Test file extension validation with complex filenames."""
        valid_names = [
            "my-document.pdf",
            "file_with_underscores.pdf",
            "document with spaces.pdf",
            "file.name.with.dots.pdf",
            "123456.pdf"
        ]

        for filename in valid_names:
            # Should not raise any exception
            PDFValidator._validate_file_extension(filename)

    def test_validation_error_messages(self, invalid_pdf_bytes, too_large_pdf_bytes, too_small_pdf_bytes):
        """Test that validation error messages contain relevant information."""
        # Test invalid format error message
        with pytest.raises(ValidationError) as exc_info:
            PDFValidator._validate_pdf_format(invalid_pdf_bytes, "test.pdf")
        assert "test.pdf" in str(exc_info.value)
        assert "not a valid PDF file" in str(exc_info.value)

        # Test file too large error message
        with pytest.raises(ValidationError) as exc_info:
            PDFValidator._validate_file_size(too_large_pdf_bytes, "large.pdf")
        assert "large.pdf" in str(exc_info.value)
        assert "too large" in str(exc_info.value)

        # Test file too small error message
        with pytest.raises(ValidationError) as exc_info:
            PDFValidator._validate_file_size(too_small_pdf_bytes, "small.pdf")
        assert "small.pdf" in str(exc_info.value)
        assert "too small" in str(exc_info.value)

    def test_validate_all_methods_called(self, sample_pdf_bytes, monkeypatch):
        """Test that validate_pdf_file calls all validation methods."""
        size_called = False
        format_called = False
        extension_called = False

        def mock_validate_file_size(pdf_bytes, filename):
            nonlocal size_called
            size_called = True

        def mock_validate_pdf_format(pdf_bytes, filename):
            nonlocal format_called
            format_called = True

        def mock_validate_file_extension(filename):
            nonlocal extension_called
            extension_called = True

        monkeypatch.setattr(PDFValidator, "_validate_file_size", mock_validate_file_size)
        monkeypatch.setattr(PDFValidator, "_validate_pdf_format", mock_validate_pdf_format)
        monkeypatch.setattr(PDFValidator, "_validate_file_extension", mock_validate_file_extension)

        PDFValidator.validate_pdf_file(sample_pdf_bytes, "test.pdf")

        assert size_called, "File size validation was not called"
        assert format_called, "PDF format validation was not called"
        assert extension_called, "File extension validation was not called"
