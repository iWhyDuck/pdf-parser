"""Tests for the processors module (legacy sync version).

This module contains comprehensive tests for PDF processing functionality
including single file processing, batch processing, and workflow orchestration.

NOTE: This file has been renamed to avoid conflicts with the new async implementation.
These tests are kept for reference but may not work with the current async architecture.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# NOTE: These imports may fail with the new async architecture
# from src.pdf_parser.processors import PDFProcessor, BatchProcessor
from src.pdf_parser.database import ExtractionRepository
from src.pdf_parser.extractors import DataExtractor, TextExtractor
from src.pdf_parser.validators import PDFValidator
from src.pdf_parser.exceptions import ValidationError, PDFProcessingError, DataExtractionError


@pytest.mark.skip(reason="Legacy sync tests - replaced by async implementation")
class TestPDFProcessor:
    """Test cases for PDFProcessor class."""

    def test_init(self, mock_repository):
        """Test PDFProcessor initialization."""
        processor = PDFProcessor(mock_repository)
        assert processor.repository == mock_repository
        assert isinstance(processor.text_extractor, TextExtractor)
        assert isinstance(processor.validator, PDFValidator)

    def test_process_file_success(self, mock_repository):
        """Test successful file processing."""
        processor = PDFProcessor(mock_repository)
        file_bytes = b"fake pdf content"
        filename = "test.pdf"
        extractor = Mock(spec=DataExtractor)
        fields = ["field1", "field2"]

        with patch.object(processor.validator, 'validate_pdf_file') as mock_validate, \
             patch.object(processor.text_extractor, 'extract_text') as mock_extract, \
             patch('hashlib.sha256') as mock_hash:

            mock_extract.return_value = "extracted text content"
            mock_hash.return_value.hexdigest.return_value = "abcdef123456"
            extractor.extract.return_value = {"field1": "value1", "field2": "value2"}

            data, file_hash = processor.process_file(file_bytes, filename, extractor, fields)

            assert data == {"field1": "value1", "field2": "value2"}
            assert file_hash == "abcdef"
            mock_validate.assert_called_once_with(file_bytes, filename)
            mock_extract.assert_called_once_with(file_bytes)
            extractor.extract.assert_called_once_with("extracted text content", fields)

    # ... rest of legacy tests would be here but marked as skipped
