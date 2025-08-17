"""Database module for the PDF parser application.

This module contains database management classes including connection
management, session creation, and repository patterns for data persistence.
"""

from typing import Any, Dict, Optional
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

from ..config import Config
from ..models import Base, Extraction
from ..exceptions import DatabaseError

__all__ = ["DatabaseManager", "ExtractionRepository"]


class DatabaseManager:
    """Manages database connections and session creation.

    This class handles SQLAlchemy engine creation, database initialization,
    and provides methods for creating database sessions. It implements
    lazy initialization for better resource management.

    Attributes:
        database_url: SQLAlchemy database URL
        _engine: Cached SQLAlchemy engine instance
        _session_factory: Cached sessionmaker factory
    """

    def __init__(self, database_url: str = Config.DATABASE_URL) -> None:
        """Initialize DatabaseManager with database URL.

        Args:
            database_url: SQLAlchemy database URL string
        """
        self.database_url: str = database_url
        self._engine: Optional[Any] = None
        self._session_factory: Optional[Any] = None

    @property
    def engine(self) -> Any:
        """Get or create SQLAlchemy engine with lazy initialization.

        Creates the engine with SQLite-specific configuration including
        connection timeout and static connection pooling.

        Returns:
            SQLAlchemy engine instance

        Raises:
            DatabaseError: If engine creation or database initialization fails
        """
        if self._engine is None:
            try:
                self._engine = create_engine(
                    self.database_url,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 20
                    },
                    poolclass=StaticPool,
                    echo=False
                )
                Base.metadata.create_all(self._engine)
            except Exception as e:
                raise DatabaseError(f"Database initialization error: {str(e)}")
        return self._engine

    def create_session(self) -> Session:
        """Create a new database session.

        Creates a new SQLAlchemy session using the configured engine.
        Each call returns a fresh session instance.

        Returns:
            SQLAlchemy Session instance
        """
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory()


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
