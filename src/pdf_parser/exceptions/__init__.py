"""Custom exceptions for the PDF parser application.

This module contains all custom exception classes used throughout
the PDF parsing and data extraction system.
"""

from .exceptions import (
    PDFProcessingError,
    DataExtractionError,
    DatabaseError,
    ValidationError
)

__all__ = [
    "PDFProcessingError",
    "DataExtractionError",
    "DatabaseError",
    "ValidationError"
]
