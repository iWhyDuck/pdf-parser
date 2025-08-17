"""Validators module for the PDF parser application.

This module contains validation classes for PDF files including
file size checks, format validation, and extension verification.
"""

from pathlib import Path

from ..config import Config
from ..exceptions import ValidationError

__all__ = ["PDFValidator"]


class PDFValidator:
    """Validates PDF files for security and correctness.

    This class provides static methods for comprehensive PDF file validation
    including file size checks, format validation, and extension verification.
    All validation methods raise appropriate exceptions for different failure modes.
    """

    @staticmethod
    def validate_pdf_file(pdf_bytes: bytes, filename: str) -> None:
        """Perform comprehensive PDF file validation.

        Executes all validation checks in sequence. If any validation
        fails, appropriate exception is raised with detailed error message.

        Args:
            pdf_bytes: Raw PDF file content as bytes
            filename: Original filename for error reporting

        Raises:
            ValidationError: If any validation check fails
        """
        PDFValidator._validate_file_size(pdf_bytes, filename)
        PDFValidator._validate_pdf_format(pdf_bytes, filename)
        PDFValidator._validate_file_extension(filename)

    @staticmethod
    def _validate_file_size(pdf_bytes: bytes, filename: str) -> None:
        """Validate PDF file size within acceptable limits.

        Checks that file size is between minimum and maximum allowed values
        to prevent processing of corrupted or excessively large files.

        Args:
            pdf_bytes: Raw PDF file content as bytes
            filename: Original filename for error reporting

        Raises:
            ValidationError: If file size is outside acceptable range
        """
        if len(pdf_bytes) > Config.MAX_FILE_SIZE:
            raise ValidationError(
                f"File {filename} is too large. Maximum size: {Config.MAX_FILE_SIZE // (1024*1024)}MB"
            )

        if len(pdf_bytes) < Config.MIN_FILE_SIZE:
            raise ValidationError(f"File {filename} is too small or corrupted")

    @staticmethod
    def _validate_pdf_format(pdf_bytes: bytes, filename: str) -> None:
        """Validate that file content is a valid PDF format.

        Checks the PDF magic number at the beginning of the file
        to ensure it's a valid PDF document.

        Args:
            pdf_bytes: Raw PDF file content as bytes
            filename: Original filename for error reporting

        Raises:
            ValidationError: If file is not a valid PDF format
        """
        if not pdf_bytes.startswith(b'%PDF'):
            raise ValidationError(f"File {filename} is not a valid PDF file")

    @staticmethod
    def _validate_file_extension(filename: str) -> None:
        """Validate that filename has correct PDF extension.

        Ensures the uploaded file has the expected .pdf extension
        to prevent processing of incorrectly named files.

        Args:
            filename: Original filename to validate

        Raises:
            ValidationError: If file extension is not .pdf
        """
        if not filename.lower().endswith('.pdf'):
            raise ValidationError(
                f"Invalid file extension. Expected .pdf, got: {Path(filename).suffix}"
            )
