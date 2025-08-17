"""Tests for the processors module.

This module contains comprehensive tests for PDF processing functionality
including single file processing, batch processing, and workflow orchestration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.pdf_parser.processors import PDFProcessor, BatchProcessor
from src.pdf_parser.database import ExtractionRepository
from src.pdf_parser.extractors import DataExtractor, TextExtractor
from src.pdf_parser.validators import PDFValidator
from src.pdf_parser.exceptions import ValidationError, PDFProcessingError, DataExtractionError


class TestPDFProcessor:
    """Test cases for PDFProcessor class."""

    def test_init(self, test_repository):
        """Test processor initialization."""
        processor = PDFProcessor(test_repository)

        assert processor.repository is test_repository
        assert isinstance(processor.text_extractor, TextExtractor)
        assert isinstance(processor.validator, PDFValidator)

    def test_process_file_success(self, test_repository, sample_pdf_bytes, sample_extraction_data):
        """Test successful file processing."""
        processor = PDFProcessor(test_repository)
        mock_extractor = Mock(spec=DataExtractor)
        mock_extractor.extract.return_value = sample_extraction_data

        with patch.object(processor.validator, 'validate_pdf_file') as mock_validate, \
             patch.object(processor.text_extractor, 'extract_text', return_value="Sample text") as mock_extract:

            data, file_hash = processor.process_file(
                sample_pdf_bytes, "test.pdf", mock_extractor, ["field1", "field2"]
            )

            assert data == sample_extraction_data
            assert len(file_hash) == 6  # SHA256 hash truncated to 6 chars

            mock_validate.assert_called_once_with(sample_pdf_bytes, "test.pdf")
            mock_extract.assert_called_once_with(sample_pdf_bytes)
            mock_extractor.extract.assert_called_once_with("Sample text", ["field1", "field2"])

    def test_process_file_validation_error(self, test_repository, sample_pdf_bytes):
        """Test handling of validation errors during file processing."""
        processor = PDFProcessor(test_repository)
        mock_extractor = Mock(spec=DataExtractor)

        with patch.object(processor.validator, 'validate_pdf_file') as mock_validate:
            mock_validate.side_effect = ValidationError("Invalid PDF")

            with pytest.raises(ValidationError, match="Invalid PDF"):
                processor.process_file(
                    sample_pdf_bytes, "test.pdf", mock_extractor, ["field1"]
                )

    def test_process_file_text_extraction_error(self, test_repository, sample_pdf_bytes):
        """Test handling of text extraction errors during file processing."""
        processor = PDFProcessor(test_repository)
        mock_extractor = Mock(spec=DataExtractor)

        with patch.object(processor.validator, 'validate_pdf_file') as mock_validate, \
             patch.object(processor.text_extractor, 'extract_text') as mock_extract:

            mock_extract.side_effect = PDFProcessingError("Text extraction failed")

            with pytest.raises(PDFProcessingError, match="Text extraction failed"):
                processor.process_file(
                    sample_pdf_bytes, "test.pdf", mock_extractor, ["field1"]
                )

    def test_process_file_data_extraction_error(self, test_repository, sample_pdf_bytes):
        """Test handling of data extraction errors during file processing."""
        processor = PDFProcessor(test_repository)
        mock_extractor = Mock(spec=DataExtractor)
        mock_extractor.extract.side_effect = DataExtractionError("Data extraction failed")

        with patch.object(processor.validator, 'validate_pdf_file') as mock_validate, \
             patch.object(processor.text_extractor, 'extract_text', return_value="Sample text") as mock_extract:

            with pytest.raises(DataExtractionError, match="Data extraction failed"):
                processor.process_file(
                    sample_pdf_bytes, "test.pdf", mock_extractor, ["field1"]
                )

    def test_process_file_hash_consistency(self, test_repository, sample_pdf_bytes):
        """Test that file hash is consistent for same file."""
        processor = PDFProcessor(test_repository)
        mock_extractor = Mock(spec=DataExtractor)
        mock_extractor.extract.return_value = {"field1": "value1"}

        with patch.object(processor.validator, 'validate_pdf_file'), \
             patch.object(processor.text_extractor, 'extract_text', return_value="Sample text"):

            _, hash1 = processor.process_file(
                sample_pdf_bytes, "test.pdf", mock_extractor, ["field1"]
            )
            _, hash2 = processor.process_file(
                sample_pdf_bytes, "test.pdf", mock_extractor, ["field1"]
            )

            assert hash1 == hash2

    def test_process_file_different_hashes_for_different_files(self, test_repository):
        """Test that different files produce different hashes."""
        processor = PDFProcessor(test_repository)
        mock_extractor = Mock(spec=DataExtractor)
        mock_extractor.extract.return_value = {"field1": "value1"}

        file1_bytes = b"PDF content 1"
        file2_bytes = b"PDF content 2"

        with patch.object(processor.validator, 'validate_pdf_file'), \
             patch.object(processor.text_extractor, 'extract_text', return_value="Sample text"):

            _, hash1 = processor.process_file(
                file1_bytes, "test1.pdf", mock_extractor, ["field1"]
            )
            _, hash2 = processor.process_file(
                file2_bytes, "test2.pdf", mock_extractor, ["field1"]
            )

            assert hash1 != hash2

    def test_save_extraction_result(self, test_repository, sample_extraction_data):
        """Test saving extraction result."""
        processor = PDFProcessor(test_repository)

        with patch.object(test_repository, 'save_extraction', return_value=123) as mock_save:
            db_id = processor.save_extraction_result(
                "test.pdf", "abc123", "classic", sample_extraction_data
            )

            assert db_id == 123
            mock_save.assert_called_once_with("test.pdf", "abc123", "classic", sample_extraction_data)


class TestBatchProcessor:
    """Test cases for BatchProcessor class."""

    def test_init(self, test_repository):
        """Test batch processor initialization."""
        pdf_processor = PDFProcessor(test_repository)
        batch_processor = BatchProcessor(pdf_processor)

        assert batch_processor.pdf_processor is pdf_processor

    def test_process_batch_success(self, test_repository, sample_extraction_data):
        """Test successful batch processing."""
        pdf_processor = PDFProcessor(test_repository)
        batch_processor = BatchProcessor(pdf_processor)

        # Mock uploaded files
        mock_file1 = Mock()
        mock_file1.name = "file1.pdf"
        mock_file1.read.return_value = b"PDF content 1"

        mock_file2 = Mock()
        mock_file2.name = "file2.pdf"
        mock_file2.read.return_value = b"PDF content 2"

        uploaded_files = [mock_file1, mock_file2]
        mock_extractor = Mock(spec=DataExtractor)
        fields = ["field1", "field2"]
        method = "classic"

        with patch.object(pdf_processor, 'process_file') as mock_process, \
             patch.object(pdf_processor, 'save_extraction_result') as mock_save, \
             patch('src.pdf_parser.processors.st') as mock_st:

            mock_process.side_effect = [
                (sample_extraction_data, "hash1"),
                (sample_extraction_data, "hash2")
            ]
            mock_save.side_effect = [10, 20]

            results = batch_processor.process_batch(
                uploaded_files, mock_extractor, fields, method
            )

            assert len(results) == 2

            # Check first result
            assert results[0]["file"] == "file1.pdf"
            assert results[0]["result"] == sample_extraction_data
            assert results[0]["db_id"] == 10

            # Check second result
            assert results[1]["file"] == "file2.pdf"
            assert results[1]["result"] == sample_extraction_data
            assert results[1]["db_id"] == 20

            # Verify calls
            assert mock_process.call_count == 2
            assert mock_save.call_count == 2
            assert mock_st.write.call_count == 2  # Success messages

    def test_process_batch_with_errors(self, test_repository, sample_extraction_data):
        """Test batch processing with some files failing."""
        pdf_processor = PDFProcessor(test_repository)
        batch_processor = BatchProcessor(pdf_processor)

        # Mock uploaded files
        mock_file1 = Mock()
        mock_file1.name = "success.pdf"
        mock_file1.read.return_value = b"PDF content 1"

        mock_file2 = Mock()
        mock_file2.name = "error.pdf"
        mock_file2.read.return_value = b"PDF content 2"

        uploaded_files = [mock_file1, mock_file2]
        mock_extractor = Mock(spec=DataExtractor)
        fields = ["field1"]
        method = "ai"

        with patch.object(pdf_processor, 'process_file') as mock_process, \
             patch.object(pdf_processor, 'save_extraction_result') as mock_save, \
             patch('src.pdf_parser.processors.st') as mock_st:

            # First file succeeds, second fails
            mock_process.side_effect = [
                (sample_extraction_data, "hash1"),
                ValidationError("Invalid PDF format")
            ]
            mock_save.return_value = 10

            results = batch_processor.process_batch(
                uploaded_files, mock_extractor, fields, method
            )

            assert len(results) == 2

            # Check successful result
            assert results[0]["file"] == "success.pdf"
            assert results[0]["result"] == sample_extraction_data
            assert results[0]["db_id"] == 10
            assert "error" not in results[0]

            # Check error result
            assert results[1]["file"] == "error.pdf"
            assert "error" in results[1]
            assert "Invalid PDF format" in results[1]["error"]
            assert "result" not in results[1]
            assert "db_id" not in results[1]

    def test_process_batch_all_files_fail(self, test_repository):
        """Test batch processing when all files fail."""
        pdf_processor = PDFProcessor(test_repository)
        batch_processor = BatchProcessor(pdf_processor)

        mock_file = Mock()
        mock_file.name = "error.pdf"
        mock_file.read.return_value = b"Invalid content"

        uploaded_files = [mock_file]
        mock_extractor = Mock(spec=DataExtractor)

        with patch.object(pdf_processor, 'process_file') as mock_process, \
             patch('src.pdf_parser.processors.st') as mock_st:

            mock_process.side_effect = Exception("Processing failed")

            results = batch_processor.process_batch(
                uploaded_files, mock_extractor, ["field1"], "classic"
            )

            assert len(results) == 1
            assert results[0]["file"] == "error.pdf"
            assert "error" in results[0]
            assert "Processing failed" in results[0]["error"]

    def test_process_batch_empty_file_list(self, test_repository):
        """Test batch processing with empty file list."""
        pdf_processor = PDFProcessor(test_repository)
        batch_processor = BatchProcessor(pdf_processor)

        mock_extractor = Mock(spec=DataExtractor)

        results = batch_processor.process_batch([], mock_extractor, ["field1"], "classic")

        assert results == []

    def test_process_batch_file_read_error(self, test_repository):
        """Test batch processing when file reading fails."""
        pdf_processor = PDFProcessor(test_repository)
        batch_processor = BatchProcessor(pdf_processor)

        mock_file = Mock()
        mock_file.name = "read_error.pdf"
        mock_file.read.side_effect = Exception("Cannot read file")

        uploaded_files = [mock_file]
        mock_extractor = Mock(spec=DataExtractor)

        with patch('src.pdf_parser.processors.st') as mock_st:
            results = batch_processor.process_batch(
                uploaded_files, mock_extractor, ["field1"], "classic"
            )

            assert len(results) == 1
            assert results[0]["file"] == "read_error.pdf"
            assert "error" in results[0]
            assert "Cannot read file" in results[0]["error"]

    def test_process_batch_save_error(self, test_repository, sample_extraction_data):
        """Test batch processing when save operation fails."""
        pdf_processor = PDFProcessor(test_repository)
        batch_processor = BatchProcessor(pdf_processor)

        mock_file = Mock()
        mock_file.name = "save_error.pdf"
        mock_file.read.return_value = b"PDF content"

        uploaded_files = [mock_file]
        mock_extractor = Mock(spec=DataExtractor)

        with patch.object(pdf_processor, 'process_file') as mock_process, \
             patch.object(pdf_processor, 'save_extraction_result') as mock_save, \
             patch('src.pdf_parser.processors.st') as mock_st:

            mock_process.return_value = (sample_extraction_data, "hash1")
            mock_save.side_effect = Exception("Database save failed")

            results = batch_processor.process_batch(
                uploaded_files, mock_extractor, ["field1"], "classic"
            )

            assert len(results) == 1
            assert results[0]["file"] == "save_error.pdf"
            assert "error" in results[0]
            assert "Database save failed" in results[0]["error"]

    def test_process_batch_progress_reporting(self, test_repository, sample_extraction_data):
        """Test that batch processing reports progress correctly."""
        pdf_processor = PDFProcessor(test_repository)
        batch_processor = BatchProcessor(pdf_processor)

        mock_files = []
        for i in range(3):
            mock_file = Mock()
            mock_file.name = f"file{i}.pdf"
            mock_file.read.return_value = f"PDF content {i}".encode()
            mock_files.append(mock_file)

        mock_extractor = Mock(spec=DataExtractor)

        with patch.object(pdf_processor, 'process_file') as mock_process, \
             patch.object(pdf_processor, 'save_extraction_result') as mock_save, \
             patch('src.pdf_parser.processors.st') as mock_st:

            mock_process.return_value = (sample_extraction_data, "hash1")
            mock_save.return_value = 1

            batch_processor.process_batch(mock_files, mock_extractor, ["field1"], "classic")

            # Should have 3 success messages
            success_calls = [call for call in mock_st.write.call_args_list
                           if "✅ Processed:" in str(call)]
            assert len(success_calls) == 3

    def test_process_batch_method_parameter_passed(self, test_repository, sample_extraction_data):
        """Test that the method parameter is correctly passed to save operation."""
        pdf_processor = PDFProcessor(test_repository)
        batch_processor = BatchProcessor(pdf_processor)

        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.read.return_value = b"PDF content"

        uploaded_files = [mock_file]
        mock_extractor = Mock(spec=DataExtractor)
        method = "ai"

        with patch.object(pdf_processor, 'process_file') as mock_process, \
             patch.object(pdf_processor, 'save_extraction_result') as mock_save, \
             patch('src.pdf_parser.processors.st'):

            mock_process.return_value = (sample_extraction_data, "hash1")
            mock_save.return_value = 10

            batch_processor.process_batch(uploaded_files, mock_extractor, ["field1"], method)

            mock_save.assert_called_once_with("test.pdf", "hash1", method, sample_extraction_data)


class TestProcessorIntegration:
    """Integration tests for processor components."""

    def test_pdf_processor_with_real_repository(self, test_repository, sample_pdf_bytes):
        """Test PDFProcessor with actual repository integration."""
        processor = PDFProcessor(test_repository)
        mock_extractor = Mock(spec=DataExtractor)
        mock_extractor.extract.return_value = {"field1": "value1"}

        with patch.object(processor.validator, 'validate_pdf_file'), \
             patch.object(processor.text_extractor, 'extract_text', return_value="Sample text"):

            # Process file
            data, file_hash = processor.process_file(
                sample_pdf_bytes, "integration.pdf", mock_extractor, ["field1"]
            )

            # Save result
            db_id = processor.save_extraction_result("integration.pdf", file_hash, "test", data)

            assert data == {"field1": "value1"}
            assert isinstance(db_id, int)
            assert db_id > 0

    def test_batch_processor_with_real_components(self, test_repository):
        """Test BatchProcessor with actual component integration."""
        pdf_processor = PDFProcessor(test_repository)
        batch_processor = BatchProcessor(pdf_processor)

        # Create mock files
        mock_files = []
        for i in range(2):
            mock_file = Mock()
            mock_file.name = f"batch_test_{i}.pdf"
            mock_file.read.return_value = f"PDF content {i}".encode()
            mock_files.append(mock_file)

        mock_extractor = Mock(spec=DataExtractor)
        mock_extractor.extract.return_value = {"test_field": "test_value"}

        with patch.object(pdf_processor.validator, 'validate_pdf_file'), \
             patch.object(pdf_processor.text_extractor, 'extract_text', return_value="Sample text"), \
             patch('src.pdf_parser.processors.st'):

            results = batch_processor.process_batch(
                mock_files, mock_extractor, ["test_field"], "integration_test"
            )

            assert len(results) == 2
            for i, result in enumerate(results):
                assert result["file"] == f"batch_test_{i}.pdf"
                assert result["result"] == {"test_field": "test_value"}
                assert isinstance(result["db_id"], int)
