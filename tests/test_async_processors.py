"""Tests for asynchronous processors in the PDF parser application.

This module contains test cases for AsyncPDFProcessor and AsyncBatchProcessor
classes, testing concurrent processing capabilities, error handling, and
progress reporting functionality.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, call, MagicMock
from typing import List, Dict, Any

from pdf_parser.processors import (
    AsyncPDFProcessor,
    AsyncBatchProcessor,
    BatchResult,
    ProgressEvent,
    ProgressEventType
)
from pdf_parser.extractors import DataExtractor
from pdf_parser.database import ExtractionRepository
from pdf_parser.exceptions import ValidationError, PDFProcessingError, DataExtractionError


@pytest.fixture
def mock_repository():
    """Create a mock extraction repository."""
    repository = Mock(spec=ExtractionRepository)
    repository.save_extraction.return_value = 123
    return repository


@pytest.fixture
def mock_extractor():
    """Create a mock data extractor."""
    extractor = Mock(spec=DataExtractor)
    extractor.extract.return_value = {"field1": "value1", "field2": "value2"}
    return extractor


@pytest.fixture
def async_pdf_processor(mock_repository):
    """Create an AsyncPDFProcessor instance with mocked dependencies."""
    processor = AsyncPDFProcessor(mock_repository)
    return processor


@pytest.fixture
def mock_uploaded_file():
    """Create a mock uploaded file object."""
    file_mock = Mock()
    file_mock.name = "test.pdf"
    file_mock.read.return_value = b"fake pdf content"
    return file_mock


@pytest.fixture
def progress_events():
    """Create a list to collect progress events."""
    events = []

    def progress_callback(event: ProgressEvent):
        events.append(event)

    return events, progress_callback


class TestAsyncPDFProcessor:
    """Test cases for AsyncPDFProcessor class."""

    @pytest.mark.asyncio
    async def test_init(self, mock_repository):
        """Test AsyncPDFProcessor initialization."""
        processor = AsyncPDFProcessor(mock_repository)
        assert processor.repository == mock_repository
        assert processor.text_extractor is not None
        assert processor.validator is not None

    @pytest.mark.asyncio
    async def test_process_file_async_success(self, async_pdf_processor, mock_extractor):
        """Test successful asynchronous file processing."""
        file_bytes = b"fake pdf content"
        filename = "test.pdf"
        fields = ["field1", "field2"]

        with patch.object(async_pdf_processor.validator, 'validate_pdf_file') as mock_validate, \
             patch.object(async_pdf_processor.text_extractor, 'extract_text') as mock_extract, \
             patch.object(async_pdf_processor, '_calculate_file_hash', return_value="abc123") as mock_hash:

            mock_extract.return_value = "extracted text content"

            data, file_hash = await async_pdf_processor.process_file_async(
                file_bytes, filename, mock_extractor, fields
            )

            # Verify all methods were called in thread pool
            assert data == {"field1": "value1", "field2": "value2"}
            assert file_hash == "abc123"
            mock_validate.assert_called_once_with(file_bytes, filename)
            mock_extract.assert_called_once_with(file_bytes)
            mock_hash.assert_called_once_with(file_bytes)
            mock_extractor.extract.assert_called_once_with("extracted text content", fields)

    @pytest.mark.asyncio
    async def test_process_file_async_validation_error(self, async_pdf_processor, mock_extractor):
        """Test file processing with validation error."""
        file_bytes = b"invalid content"
        filename = "invalid.pdf"
        fields = ["field1"]

        with patch.object(async_pdf_processor.validator, 'validate_pdf_file') as mock_validate:
            mock_validate.side_effect = ValidationError("Invalid PDF file")

            with pytest.raises(ValidationError, match="Invalid PDF file"):
                await async_pdf_processor.process_file_async(
                    file_bytes, filename, mock_extractor, fields
                )

    @pytest.mark.asyncio
    async def test_process_file_async_text_extraction_error(self, async_pdf_processor, mock_extractor):
        """Test file processing with text extraction error."""
        file_bytes = b"corrupted pdf"
        filename = "corrupted.pdf"
        fields = ["field1"]

        with patch.object(async_pdf_processor.validator, 'validate_pdf_file'), \
             patch.object(async_pdf_processor.text_extractor, 'extract_text') as mock_extract:

            mock_extract.side_effect = PDFProcessingError("Text extraction failed")

            with pytest.raises(PDFProcessingError, match="Text extraction failed"):
                await async_pdf_processor.process_file_async(
                    file_bytes, filename, mock_extractor, fields
                )

    @pytest.mark.asyncio
    async def test_process_file_async_data_extraction_error(self, async_pdf_processor, mock_extractor):
        """Test file processing with data extraction error."""
        file_bytes = b"valid pdf"
        filename = "test.pdf"
        fields = ["nonexistent_field"]

        with patch.object(async_pdf_processor.validator, 'validate_pdf_file'), \
             patch.object(async_pdf_processor.text_extractor, 'extract_text') as mock_extract, \
             patch.object(async_pdf_processor, '_calculate_file_hash', return_value="abc123"):

            mock_extract.return_value = "extracted text"
            mock_extractor.extract.side_effect = DataExtractionError("Field not found")

            with pytest.raises(DataExtractionError, match="Field not found"):
                await async_pdf_processor.process_file_async(
                    file_bytes, filename, mock_extractor, fields
                )

    @pytest.mark.asyncio
    async def test_save_extraction_result_async(self, async_pdf_processor):
        """Test asynchronous saving of extraction results."""
        filename = "test.pdf"
        file_hash = "abc123"
        method = "ai"
        data = {"field1": "value1"}

        result = await async_pdf_processor.save_extraction_result_async(
            filename, file_hash, method, data
        )

        assert result == 123
        async_pdf_processor.repository.save_extraction.assert_called_once_with(
            filename, file_hash, method, data
        )

    def test_calculate_file_hash(self, async_pdf_processor):
        """Test file hash calculation."""
        file_bytes = b"test content"
        hash_result = async_pdf_processor._calculate_file_hash(file_bytes)

        assert len(hash_result) == 6
        assert isinstance(hash_result, str)

        # Same content should produce same hash
        hash_result2 = async_pdf_processor._calculate_file_hash(file_bytes)
        assert hash_result == hash_result2

        # Different content should produce different hash
        different_bytes = b"different content"
        different_hash = async_pdf_processor._calculate_file_hash(different_bytes)
        assert hash_result != different_hash


class TestAsyncBatchProcessor:
    """Test cases for AsyncBatchProcessor class."""

    @pytest.mark.asyncio
    async def test_init(self, async_pdf_processor):
        """Test AsyncBatchProcessor initialization."""
        progress_callback = Mock()
        processor = AsyncBatchProcessor(
            async_pdf_processor,
            max_concurrent=3,
            progress_callback=progress_callback
        )

        assert processor.pdf_processor == async_pdf_processor
        assert processor.max_concurrent == 3
        assert processor.progress_callback == progress_callback
        assert processor._semaphore._value == 3

    @pytest.mark.asyncio
    async def test_process_batch_success(self, async_pdf_processor, mock_extractor, progress_events):
        """Test successful batch processing of multiple files."""
        events, progress_callback = progress_events

        # Create mock files
        files = []
        for i in range(3):
            file_mock = Mock()
            file_mock.name = f"test{i}.pdf"
            file_mock.read.return_value = f"content{i}".encode()
            files.append(file_mock)

        processor = AsyncBatchProcessor(
            async_pdf_processor,
            max_concurrent=2,
            progress_callback=progress_callback
        )

        # Mock the async methods
        with patch.object(async_pdf_processor, 'process_file_async', new_callable=AsyncMock) as mock_process, \
             patch.object(async_pdf_processor, 'save_extraction_result_async', new_callable=AsyncMock) as mock_save:

            mock_process.return_value = ({"field": "value"}, "hash123")
            mock_save.return_value = 456

            results = await processor.process_batch(
                files, mock_extractor, ["field1"], "ai"
            )

        # Verify results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, BatchResult)
            assert result.file_name == f"test{i}.pdf"
            assert result.success is True
            assert result.data == {"field": "value"}
            assert result.db_id == 456
            assert result.error is None

        # Verify progress events
        assert len(events) >= 2  # At least BATCH_STARTED and BATCH_COMPLETED
        assert events[0].event_type == ProgressEventType.BATCH_STARTED
        assert events[-1].event_type == ProgressEventType.BATCH_COMPLETED
        assert events[0].total_files == 3
        assert "3 files" in events[0].message

    @pytest.mark.asyncio
    async def test_process_batch_with_errors(self, async_pdf_processor, mock_extractor, progress_events):
        """Test batch processing with some files failing."""
        events, progress_callback = progress_events

        # Create mock files
        files = []
        for i in range(3):
            file_mock = Mock()
            file_mock.name = f"test{i}.pdf"
            file_mock.read.return_value = f"content{i}".encode()
            files.append(file_mock)

        processor = AsyncBatchProcessor(
            async_pdf_processor,
            progress_callback=progress_callback
        )

        # Mock the async methods with mixed success/failure
        with patch.object(async_pdf_processor, 'process_file_async', new_callable=AsyncMock) as mock_process, \
             patch.object(async_pdf_processor, 'save_extraction_result_async', new_callable=AsyncMock) as mock_save:

            async def process_side_effect(file_bytes, filename, extractor, fields):
                if "test1" in filename:
                    raise PDFProcessingError("Processing failed")
                return ({"field": "value"}, "hash123")

            mock_process.side_effect = process_side_effect
            mock_save.return_value = 789

            results = await processor.process_batch(
                files, mock_extractor, ["field1"], "ai"
            )

        # Verify results
        assert len(results) == 3

        # Check successful files
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        assert len(successful_results) == 2
        assert len(failed_results) == 1

        # Check failed result
        failed_result = failed_results[0]
        assert failed_result.file_name == "test1.pdf"
        assert failed_result.success is False
        assert failed_result.error == "Processing failed"
        assert failed_result.data is None
        assert failed_result.db_id is None

        # Verify progress events include failures
        failed_events = [e for e in events if e.event_type == ProgressEventType.FILE_FAILED]
        assert len(failed_events) == 1
        assert failed_events[0].file_name == "test1.pdf"
        assert "Processing failed" in failed_events[0].error

    @pytest.mark.asyncio
    async def test_process_batch_all_files_fail(self, async_pdf_processor, mock_extractor, progress_events):
        """Test batch processing when all files fail."""
        events, progress_callback = progress_events

        files = [Mock(name="test1.pdf", read=Mock(return_value=b"content1")),
                Mock(name="test2.pdf", read=Mock(return_value=b"content2"))]

        processor = AsyncBatchProcessor(
            async_pdf_processor,
            progress_callback=progress_callback
        )

        with patch.object(async_pdf_processor, 'process_file_async', new_callable=AsyncMock) as mock_process:
            mock_process.side_effect = ValidationError("All files invalid")

            results = await processor.process_batch(
                files, mock_extractor, ["field1"], "classic"
            )

        # All results should be failures
        assert len(results) == 2
        assert all(not r.success for r in results)
        assert all(r.error == "All files invalid" for r in results)

        # Check completion message indicates all failures
        completion_event = events[-1]
        assert completion_event.event_type == ProgressEventType.BATCH_COMPLETED
        assert "0 successful, 2 failed" in completion_event.message

    @pytest.mark.asyncio
    async def test_process_batch_empty_file_list(self, async_pdf_processor, mock_extractor, progress_events):
        """Test batch processing with empty file list."""
        events, progress_callback = progress_events

        processor = AsyncBatchProcessor(
            async_pdf_processor,
            progress_callback=progress_callback
        )

        results = await processor.process_batch([], mock_extractor, ["field1"], "ai")

        assert len(results) == 0
        assert len(events) == 2  # BATCH_STARTED and BATCH_COMPLETED
        assert events[0].total_files == 0
        assert "0 files" in events[0].message

    @pytest.mark.asyncio
    async def test_process_batch_with_progress_alternative(self, async_pdf_processor, mock_extractor, progress_events):
        """Test alternative process_batch_with_progress method."""
        events, progress_callback = progress_events

        files = [Mock(name="test1.pdf", read=Mock(return_value=b"content1")),
                Mock(name="test2.pdf", read=Mock(return_value=b"content2"))]

        processor = AsyncBatchProcessor(
            async_pdf_processor,
            progress_callback=progress_callback
        )

        with patch.object(async_pdf_processor, 'process_file_async', new_callable=AsyncMock) as mock_process, \
             patch.object(async_pdf_processor, 'save_extraction_result_async', new_callable=AsyncMock) as mock_save:

            mock_process.return_value = ({"field": "value"}, "hash123")
            mock_save.return_value = 999

            results = await processor.process_batch_with_progress(
                files, mock_extractor, ["field1"], "ai"
            )

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.db_id == 999 for r in results)

    @pytest.mark.asyncio
    async def test_read_file_async_with_sync_fallback(self, async_pdf_processor):
        """Test asynchronous file reading with sync fallback."""
        processor = AsyncBatchProcessor(async_pdf_processor)

        # Test with sync file object
        sync_file = Mock()
        sync_file.read.return_value = b"sync content"

        content = await processor._read_file_async(sync_file)
        assert content == b"sync content"

    @pytest.mark.asyncio
    async def test_read_file_async_with_async_file(self, async_pdf_processor):
        """Test asynchronous file reading with async file object."""
        processor = AsyncBatchProcessor(async_pdf_processor)

        # Test with async file object
        async_file = Mock()
        async_file.aread = AsyncMock(return_value=b"async content")

        content = await processor._read_file_async(async_file)
        assert content == b"async content"
        async_file.aread.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrency_limiting(self, async_pdf_processor, mock_extractor):
        """Test that concurrency is properly limited by semaphore."""
        max_concurrent = 2
        processor = AsyncBatchProcessor(
            async_pdf_processor,
            max_concurrent=max_concurrent
        )

        # Create many files to test concurrency limiting
        files = []
        for i in range(5):
            file_mock = Mock()
            file_mock.name = f"test{i}.pdf"
            file_mock.read.return_value = f"content{i}".encode()
            files.append(file_mock)

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent_seen = 0

        async def mock_process_with_delay(*args):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.01)  # Simulate processing time
            concurrent_count -= 1
            return ({"field": "value"}, "hash123")

        with patch.object(async_pdf_processor, 'process_file_async', side_effect=mock_process_with_delay), \
             patch.object(async_pdf_processor, 'save_extraction_result_async', new_callable=AsyncMock, return_value=123):

            results = await processor.process_batch(
                files, mock_extractor, ["field1"], "ai"
            )

        # Verify concurrency was limited
        assert max_concurrent_seen <= max_concurrent
        assert len(results) == 5
        assert all(r.success for r in results)

    def test_emit_progress_with_callback(self, async_pdf_processor):
        """Test progress event emission with callback."""
        events = []

        def progress_callback(event: ProgressEvent):
            events.append(event)

        processor = AsyncBatchProcessor(
            async_pdf_processor,
            progress_callback=progress_callback
        )

        test_event = ProgressEvent(
            event_type=ProgressEventType.FILE_STARTED,
            file_name="test.pdf",
            message="Test message"
        )

        processor._emit_progress(test_event)

        assert len(events) == 1
        assert events[0] == test_event

    def test_emit_progress_without_callback(self, async_pdf_processor):
        """Test progress event emission without callback (should not crash)."""
        processor = AsyncBatchProcessor(async_pdf_processor)  # No callback

        test_event = ProgressEvent(
            event_type=ProgressEventType.FILE_STARTED,
            file_name="test.pdf",
            message="Test message"
        )

        # Should not raise an exception
        processor._emit_progress(test_event)


class TestProgressEvent:
    """Test cases for ProgressEvent data class."""

    def test_progress_event_creation(self):
        """Test ProgressEvent creation with various parameters."""
        event = ProgressEvent(
            event_type=ProgressEventType.FILE_STARTED,
            file_name="test.pdf",
            current_file=1,
            total_files=5,
            message="Processing started",
            error=None
        )

        assert event.event_type == ProgressEventType.FILE_STARTED
        assert event.file_name == "test.pdf"
        assert event.current_file == 1
        assert event.total_files == 5
        assert event.message == "Processing started"
        assert event.error is None

    def test_progress_event_with_error(self):
        """Test ProgressEvent creation with error information."""
        event = ProgressEvent(
            event_type=ProgressEventType.FILE_FAILED,
            file_name="failed.pdf",
            error="Processing failed"
        )

        assert event.event_type == ProgressEventType.FILE_FAILED
        assert event.file_name == "failed.pdf"
        assert event.error == "Processing failed"


class TestBatchResult:
    """Test cases for BatchResult data class."""

    def test_successful_batch_result(self):
        """Test creation of successful BatchResult."""
        result = BatchResult(
            file_name="test.pdf",
            success=True,
            data={"field1": "value1"},
            db_id=123
        )

        assert result.file_name == "test.pdf"
        assert result.success is True
        assert result.data == {"field1": "value1"}
        assert result.db_id == 123
        assert result.error is None

    def test_failed_batch_result(self):
        """Test creation of failed BatchResult."""
        result = BatchResult(
            file_name="failed.pdf",
            success=False,
            error="Processing failed"
        )

        assert result.file_name == "failed.pdf"
        assert result.success is False
        assert result.data is None
        assert result.db_id is None
        assert result.error == "Processing failed"


class TestAsyncProcessorIntegration:
    """Integration tests for async processors with real components."""

    @pytest.mark.asyncio
    async def test_end_to_end_batch_processing(self, mock_repository, mock_extractor):
        """Test end-to-end async batch processing with real workflow."""
        # Create processor chain
        async_pdf_processor = AsyncPDFProcessor(mock_repository)

        events = []
        def progress_callback(event: ProgressEvent):
            events.append(event)

        batch_processor = AsyncBatchProcessor(
            async_pdf_processor,
            max_concurrent=2,
            progress_callback=progress_callback
        )

        # Create test files
        files = []
        for i in range(3):
            file_mock = Mock()
            file_mock.name = f"document{i}.pdf"
            file_mock.read.return_value = f"PDF content {i}".encode()
            files.append(file_mock)

        # Mock all the dependencies
        with patch.object(async_pdf_processor.validator, 'validate_pdf_file'), \
             patch.object(async_pdf_processor.text_extractor, 'extract_text', return_value="extracted text"), \
             patch.object(async_pdf_processor, '_calculate_file_hash', return_value="hash123"), \
             patch.object(async_pdf_processor.repository, 'save_extraction', return_value=123):

            results = await batch_processor.process_batch(
                files, mock_extractor, ["name", "date"], "ai"
            )

        # Verify end-to-end results
        assert len(results) == 3
        assert all(isinstance(r, BatchResult) for r in results)
        assert all(r.success for r in results)
        assert all(r.data == {"field1": "value1", "field2": "value2"} for r in results)
        assert all(r.db_id == 123 for r in results)

        # Verify progress tracking
        assert len(events) >= 8  # START + 3*(FILE_START + FILE_COMPLETE) + COMPLETE
        start_events = [e for e in events if e.event_type == ProgressEventType.BATCH_STARTED]
        complete_events = [e for e in events if e.event_type == ProgressEventType.BATCH_COMPLETED]
        file_complete_events = [e for e in events if e.event_type == ProgressEventType.FILE_COMPLETED]

        assert len(start_events) == 1
        assert len(complete_events) == 1
        assert len(file_complete_events) == 3

        assert start_events[0].total_files == 3
        assert "3 successful, 0 failed" in complete_events[0].message
