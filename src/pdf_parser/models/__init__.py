"""Database models for the PDF parser application.

This module contains SQLAlchemy model definitions for storing
PDF extraction results and related data.
"""

from sqlalchemy import Column, DateTime, Integer, String, Text, func
from sqlalchemy.orm import declarative_base

__all__ = ["Base", "Extraction"]

Base = declarative_base()


class Extraction(Base):
    """SQLAlchemy model for storing PDF extraction results.

    This model represents a single PDF extraction operation result,
    including metadata about the file, extraction method used,
    and the extracted data in JSON format.

    Attributes:
        id: Primary key for the extraction record
        filename: Original name of the processed PDF file
        file_hash: SHA256 hash of the file content (first 6 characters)
        extraction_method: Method used for extraction ('classic' or 'ai')
        extracted_data: JSON string containing the extracted field-value pairs
        created_at: Timestamp of when the extraction was performed
    """
    __tablename__ = "extractions"

    id: int = Column(Integer, primary_key=True)
    filename: str = Column(String(255), nullable=False)
    file_hash: str = Column(String(64), nullable=False)
    extraction_method: str = Column(String(50), nullable=False)
    extracted_data: str = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
