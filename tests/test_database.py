"""Tests for the database module.

This module contains comprehensive tests for database management functionality
including database connection management, session creation, and repository operations.
"""

import json
import pytest
from unittest.mock import Mock, patch
from sqlalchemy.exc import SQLAlchemyError

from src.pdf_parser.database import DatabaseManager, ExtractionRepository
from src.pdf_parser.models import Extraction
from src.pdf_parser.exceptions import DatabaseError
from src.pdf_parser.config import Config


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""

    def test_init_default_url(self):
        """Test initialization with default database URL."""
        db_manager = DatabaseManager()
        assert db_manager.database_url == Config.DATABASE_URL
        assert db_manager._engine is None
        assert db_manager._session_factory is None

    def test_init_custom_url(self):
        """Test initialization with custom database URL."""
        custom_url = "sqlite:///test.db"
        db_manager = DatabaseManager(custom_url)
        assert db_manager.database_url == custom_url

    def test_engine_property_lazy_initialization(self, test_db_manager):
        """Test that engine is created lazily and cached."""
        # Engine should be None initially
        assert test_db_manager._engine is None

        # First access should create the engine
        engine = test_db_manager.engine
        assert engine is not None
        assert test_db_manager._engine is engine

        # Second access should return the same engine
        engine2 = test_db_manager.engine
        assert engine is engine2

    def test_engine_property_database_creation(self, test_db_manager):
        """Test that engine creation also creates database tables."""
        with patch('src.pdf_parser.database.database_manager.Base.metadata.create_all') as mock_create_all:
            engine = test_db_manager.engine
            mock_create_all.assert_called_once_with(engine)

    def test_engine_property_creation_error(self):
        """Test handling of engine creation errors."""
        with patch('src.pdf_parser.database.database_manager.create_engine') as mock_create_engine:
            mock_create_engine.side_effect = Exception("Database connection failed")

            db_manager = DatabaseManager("invalid://url")

            with pytest.raises(DatabaseError, match="Database initialization error"):
                _ = db_manager.engine

    def test_engine_configuration(self, test_db_manager):
        """Test that engine is configured with correct parameters."""
        with patch('src.pdf_parser.database.database_manager.create_engine') as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine

            _ = test_db_manager.engine

            mock_create_engine.assert_called_once()
            args, kwargs = mock_create_engine.call_args

            assert args[0] == test_db_manager.database_url
            assert kwargs['connect_args']['check_same_thread'] is False
            assert kwargs['connect_args']['timeout'] == 20
            assert kwargs['echo'] is False

    def test_create_session_success(self, test_db_manager):
        """Test successful session creation."""
        session = test_db_manager.create_session()
        assert session is not None
        session.close()

    def test_create_session_factory_caching(self, test_db_manager):
        """Test that session factory is created and cached."""
        # Session factory should be None initially
        assert test_db_manager._session_factory is None

        # First session creation should create the factory
        session1 = test_db_manager.create_session()
        assert test_db_manager._session_factory is not None

        # Second session creation should reuse the factory
        session_factory = test_db_manager._session_factory
        session2 = test_db_manager.create_session()
        assert test_db_manager._session_factory is session_factory

        session1.close()
        session2.close()

    def test_create_session_multiple_sessions(self, test_db_manager):
        """Test that multiple sessions can be created independently."""
        session1 = test_db_manager.create_session()
        session2 = test_db_manager.create_session()

        assert session1 is not session2
        assert session1 is not None
        assert session2 is not None

        session1.close()
        session2.close()


class TestExtractionRepository:
    """Test cases for ExtractionRepository class."""

    def test_init(self, test_db_manager):
        """Test repository initialization."""
        repo = ExtractionRepository(test_db_manager)
        assert repo.db_manager is test_db_manager

    def test_save_extraction_success(self, test_repository, sample_extraction_data):
        """Test successful extraction saving."""
        filename = "test.pdf"
        file_hash = "abc123"
        method = "classic"

        db_id = test_repository.save_extraction(filename, file_hash, method, sample_extraction_data)

        assert db_id is not None
        assert isinstance(db_id, int)
        assert db_id > 0

    def test_save_extraction_data_serialization(self, test_repository):
        """Test that extraction data is properly serialized to JSON."""
        filename = "test.pdf"
        file_hash = "abc123"
        method = "ai"
        data = {
            "field1": "value1",
            "field2": "value2",
            "unicode_field": "café"
        }

        with patch.object(test_repository.db_manager, 'create_session') as mock_create_session:
            mock_session = Mock()
            mock_create_session.return_value = mock_session

            # Create a mock record with the expected data
            with patch('src.pdf_parser.database.extraction_repository.Extraction') as mock_extraction_class:
                mock_record = Mock()
                mock_record.id = 1
                mock_extraction_class.return_value = mock_record

                db_id = test_repository.save_extraction(filename, file_hash, method, data)

                # Verify Extraction was called with correct arguments
                mock_extraction_class.assert_called_once_with(
                    filename=filename,
                    file_hash=file_hash,
                    extraction_method=method,
                    extracted_data=json.dumps(data, ensure_ascii=False)
                )

                mock_session.add.assert_called_once()
                mock_session.commit.assert_called_once()
                mock_session.close.assert_called_once()

    def test_save_extraction_database_error(self, test_repository, sample_extraction_data):
        """Test handling of database errors during save."""
        with patch.object(test_repository.db_manager, 'create_session') as mock_create_session:
            mock_session = Mock()
            mock_create_session.return_value = mock_session
            mock_session.commit.side_effect = SQLAlchemyError("Database error")

            with pytest.raises(DatabaseError, match="Database save error"):
                test_repository.save_extraction("test.pdf", "abc123", "classic", sample_extraction_data)

            mock_session.rollback.assert_called_once()
            mock_session.close.assert_called_once()

    def test_save_extraction_session_cleanup_on_success(self, test_repository, sample_extraction_data):
        """Test that session is properly closed on successful save."""
        with patch.object(test_repository.db_manager, 'create_session') as mock_create_session:
            mock_session = Mock()
            mock_record = Mock()
            mock_record.id = 1
            mock_create_session.return_value = mock_session

            with patch('src.pdf_parser.database.extraction_repository.Extraction') as mock_extraction_class:
                mock_extraction_class.return_value = mock_record

                test_repository.save_extraction("test.pdf", "abc123", "classic", sample_extraction_data)

                mock_session.close.assert_called_once()

    def test_save_extraction_session_cleanup_on_error(self, test_repository, sample_extraction_data):
        """Test that session is properly closed even when error occurs."""
        with patch.object(test_repository.db_manager, 'create_session') as mock_create_session:
            mock_session = Mock()
            mock_create_session.return_value = mock_session

            with patch('src.pdf_parser.database.extraction_repository.Extraction') as mock_extraction_class:
                mock_extraction_class.side_effect = SQLAlchemyError("Unexpected error")

                with pytest.raises(DatabaseError):
                    test_repository.save_extraction("test.pdf", "abc123", "classic", sample_extraction_data)

                mock_session.rollback.assert_called_once()
                mock_session.close.assert_called_once()

    def test_save_extraction_record_creation(self, test_repository, sample_extraction_data):
        """Test that Extraction record is created with correct attributes."""
        filename = "document.pdf"
        file_hash = "def456"
        method = "ai"

        with patch.object(test_repository.db_manager, 'create_session') as mock_create_session:
            mock_session = Mock()
            mock_create_session.return_value = mock_session

            with patch('src.pdf_parser.database.extraction_repository.Extraction') as mock_extraction_class:
                mock_record = Mock()
                mock_record.id = 42
                mock_extraction_class.return_value = mock_record

                db_id = test_repository.save_extraction(filename, file_hash, method, sample_extraction_data)

                # Verify Extraction was called with correct arguments
                mock_extraction_class.assert_called_once()
                call_kwargs = mock_extraction_class.call_args.kwargs

                assert call_kwargs['filename'] == filename
                assert call_kwargs['file_hash'] == file_hash
                assert call_kwargs['extraction_method'] == method
                assert call_kwargs['extracted_data'] == json.dumps(sample_extraction_data, ensure_ascii=False)
                assert db_id == 42

    def test_save_extraction_unicode_handling(self, test_repository):
        """Test proper handling of Unicode characters in extraction data."""
        data_with_unicode = {
            "name": "José García",
            "address": "123 Café Street",
            "note": "Special chars: àáâãäåæçèéêë"
        }

        with patch.object(test_repository.db_manager, 'create_session') as mock_create_session:
            mock_session = Mock()
            mock_create_session.return_value = mock_session

            with patch('src.pdf_parser.database.extraction_repository.Extraction') as mock_extraction_class:
                mock_record = Mock()
                mock_record.id = 1
                mock_extraction_class.return_value = mock_record

                test_repository.save_extraction("test.pdf", "abc123", "classic", data_with_unicode)

                # Verify that ensure_ascii=False is used for proper Unicode handling
                call_kwargs = mock_extraction_class.call_args.kwargs
                expected_json = json.dumps(data_with_unicode, ensure_ascii=False)
                assert call_kwargs['extracted_data'] == expected_json

    def test_save_extraction_empty_data(self, test_repository):
        """Test saving extraction with empty data."""
        empty_data = {}

        db_id = test_repository.save_extraction("empty.pdf", "empty123", "classic", empty_data)

        assert db_id is not None
        assert isinstance(db_id, int)

    def test_save_extraction_complex_data(self, test_repository):
        """Test saving extraction with complex nested data."""
        complex_data = {
            "simple_field": "value",
            "nested_dict": {
                "inner_field": "inner_value",
                "numbers": [1, 2, 3]
            },
            "list_field": ["item1", "item2"],
            "null_field": None,
            "boolean_field": True
        }

        db_id = test_repository.save_extraction("complex.pdf", "complex123", "ai", complex_data)

        assert db_id is not None
        assert isinstance(db_id, int)


class TestDatabaseIntegration:
    """Integration tests for database components."""

    def test_full_workflow(self, test_db_manager, sample_extraction_data):
        """Test complete database workflow from manager to repository."""
        # Create repository
        repo = ExtractionRepository(test_db_manager)

        # Save extraction
        db_id = repo.save_extraction("integration.pdf", "int123", "classic", sample_extraction_data)

        # Verify record was saved
        session = test_db_manager.create_session()
        try:
            record = session.query(Extraction).filter_by(id=db_id).first()
            assert record is not None
            assert record.filename == "integration.pdf"
            assert record.file_hash == "int123"
            assert record.extraction_method == "classic"

            # Verify data can be deserialized
            saved_data = json.loads(record.extracted_data)
            assert saved_data == sample_extraction_data

        finally:
            session.close()

    def test_multiple_extractions(self, test_db_manager):
        """Test saving multiple extractions."""
        repo = ExtractionRepository(test_db_manager)

        # Save multiple records
        data1 = {"field1": "value1"}
        data2 = {"field2": "value2"}

        db_id1 = repo.save_extraction("file1.pdf", "hash1", "classic", data1)
        db_id2 = repo.save_extraction("file2.pdf", "hash2", "ai", data2)

        assert db_id1 != db_id2

        # Verify both records exist
        session = test_db_manager.create_session()
        try:
            record1 = session.query(Extraction).filter_by(id=db_id1).first()
            record2 = session.query(Extraction).filter_by(id=db_id2).first()

            assert record1.filename == "file1.pdf"
            assert record2.filename == "file2.pdf"
            assert record1.extraction_method == "classic"
            assert record2.extraction_method == "ai"

        finally:
            session.close()

    def test_database_persistence(self, test_db_manager, sample_extraction_data):
        """Test that data persists across database manager instances."""
        # Save data with first instance
        repo1 = ExtractionRepository(test_db_manager)
        db_id = repo1.save_extraction("persist.pdf", "persist123", "classic", sample_extraction_data)

        # Create new database manager with same URL
        db_manager2 = DatabaseManager(test_db_manager.database_url)

        # Verify data is accessible with new instance
        session = db_manager2.create_session()
        try:
            record = session.query(Extraction).filter_by(id=db_id).first()
            assert record is not None
            assert record.filename == "persist.pdf"

        finally:
            session.close()
