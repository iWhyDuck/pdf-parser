"""Database manager for the PDF parser application.

This module contains the DatabaseManager class for handling database
connections, session creation, and database initialization.
"""

from typing import Any, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ..config import Config
from ..models import Base
from ..exceptions import DatabaseError

__all__ = ["DatabaseManager"]


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
