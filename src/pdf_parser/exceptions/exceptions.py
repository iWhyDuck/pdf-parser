"""Custom exceptions for the PDF parser application.

This module contains all custom exception classes used throughout
the PDF parsing and data extraction system.
"""

__all__ = [
    "PDFProcessingError",
    "DataExtractionError",
    "DatabaseError",
    "ValidationError"
]


class PDFProcessingError(Exception):
    """Exception raised during PDF file processing operations.

    This exception is raised when there are issues with PDF file reading,
    parsing, or text extraction.
    """
    pass


class DataExtractionError(Exception):
    """Exception raised during data extraction operations.

    This exception is raised when there are issues with regex matching,
    AI processing, or field value extraction.
    """
    pass


class DatabaseError(Exception):
    """Exception raised during database operations.

    This exception is raised when there are issues with database
    connectivity, queries, or data persistence.
    """
    pass


class ValidationError(Exception):
    """Exception raised during data validation.

    This exception is raised when input data fails validation checks
    such as file format, size, or content validation.
    """
    pass
