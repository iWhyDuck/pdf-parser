"""Extraction repository for the PDF parser application.

This module contains the ExtractionRepository class for handling
extraction data persistence using the repository pattern.
"""

from typing import Any, Dict
import json

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..models import Extraction
from ..exceptions import DatabaseError
from .database_manager import DatabaseManager

__all__ = ["ExtractionRepository"]


class ExtractionRepository:
    """Repository pattern implementation for extraction data persistence.

    This class encapsulates all database operations related to extraction
    records, providing a clean interface for data persistence operations.

    Attributes:
        db_manager: DatabaseManager instance for database operations
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize repository with database manager.

        Args:
            db_manager: DatabaseManager instance for database operations
        """
        self.db_manager: DatabaseManager = db_manager

    def save_extraction(self, filename: str, file_hash: str,
                       method: str, data: Dict[str, Any]) -> int:
        """Save extraction result to database.

        Persists the extraction result as a new record in the database.
        Handles session management and error recovery automatically.

        Args:
            filename: Original PDF filename
            file_hash: SHA256 hash of the file content
            method: Extraction method used ('classic' or 'ai')
            data: Dictionary containing extracted field-value pairs

        Returns:
            Database ID of the created extraction record

        Raises:
            DatabaseError: If database operation fails
        """
        session: Session = self.db_manager.create_session()
        try:
            record = Extraction(
                filename=filename,
                file_hash=file_hash,
                extraction_method=method,
                extracted_data=json.dumps(data, ensure_ascii=False)
            )
            session.add(record)
            session.commit()
            return record.id
        except SQLAlchemyError as e:
            session.rollback()
            raise DatabaseError(f"Database save error: {str(e)}")
        finally:
            session.close()
