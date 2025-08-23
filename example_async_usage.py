"""Example usage of AsyncBatchProcessor for concurrent PDF processing.

This script demonstrates how to use the asynchronous batch processor
to process multiple PDF files concurrently with progress tracking.
"""

import asyncio
import time
from pathlib import Path
from typing import List
from unittest.mock import Mock

from src.pdf_parser.processors import AsyncPDFProcessor, AsyncBatchProcessor, ProgressEvent, ProgressEventType
from src.pdf_parser.extractors import ClassicExtractor
from src.pdf_parser.database import DatabaseManager, ExtractionRepository


class MockFile:
    """Mock file object for demonstration."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def read(self) -> bytes:
        """Read file content."""
        return self._content


def progress_handler(event: ProgressEvent) -> None:
    """Handle progress events with colored output."""
    timestamp = time.strftime("%H:%M:%S")

    if event.event_type == ProgressEventType.BATCH_STARTED:
        print(f"🚀 [{timestamp}] Starting batch processing of {event.total_files} files")

    elif event.event_type == ProgressEventType.FILE_STARTED:
        print(f"📄 [{timestamp}] Processing {event.file_name} ({event.current_file}/{event.total_files})")

    elif event.event_type == ProgressEventType.FILE_COMPLETED:
        print(f"✅ [{timestamp}] Completed {event.file_name} ({event.current_file}/{event.total_files})")

    elif event.event_type == ProgressEventType.FILE_FAILED:
        print(f"❌ [{timestamp}] Failed {event.file_name}: {event.error}")

    elif event.event_type == ProgressEventType.BATCH_COMPLETED:
        print(f"🎉 [{timestamp}] {event.message}")


async def demonstrate_async_batch_processing():
    """Demonstrate async batch processing with mock data."""
    print("=== Async PDF Batch Processing Demo ===\n")

    # Create mock PDF files
    mock_files = [
        MockFile("invoice_001.pdf", b"Mock PDF content for invoice 001"),
        MockFile("invoice_002.pdf", b"Mock PDF content for invoice 002"),
        MockFile("invoice_003.pdf", b"Mock PDF content for invoice 003"),
        MockFile("invoice_004.pdf", b"Mock PDF content for invoice 004"),
        MockFile("invoice_005.pdf", b"Mock PDF content for invoice 005"),
    ]

    # Setup database and repository
    db_manager = DatabaseManager()
    repository = ExtractionRepository(db_manager)

    # Create processors
    async_pdf_processor = AsyncPDFProcessor(repository)
    batch_processor = AsyncBatchProcessor(
        pdf_processor=async_pdf_processor,
        max_concurrent=3,  # Process max 3 files concurrently
        progress_callback=progress_handler
    )

    # Create extractor
    extractor = ClassicExtractor()
    fields_to_extract = ["invoice_number", "date", "total_amount", "vendor_name"]

    print("Configuration:")
    print(f"  - Files to process: {len(mock_files)}")
    print(f"  - Max concurrent: 3")
    print(f"  - Fields to extract: {', '.join(fields_to_extract)}")
    print(f"  - Extraction method: Classic\n")

    # Record start time
    start_time = time.time()

    try:
        # Process files concurrently
        results = await batch_processor.process_batch(
            uploaded_files=mock_files,
            extractor=extractor,
            fields=fields_to_extract,
            method="classic"
        )

        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time

        # Display results
        print(f"\n=== Processing Results ===")
        print(f"Total processing time: {processing_time:.2f} seconds")
        print(f"Average time per file: {processing_time/len(mock_files):.2f} seconds\n")

        successful_files = [r for r in results if r.success]
        failed_files = [r for r in results if not r.success]

        print(f"✅ Successfully processed: {len(successful_files)} files")
        for result in successful_files:
            print(f"   - {result.file_name} (DB ID: {result.db_id})")

        if failed_files:
            print(f"\n❌ Failed to process: {len(failed_files)} files")
            for result in failed_files:
                print(f"   - {result.file_name}: {result.error}")

        # Show sample extracted data
        if successful_files:
            print(f"\n=== Sample Extracted Data ===")
            sample_result = successful_files[0]
            print(f"File: {sample_result.file_name}")
            if sample_result.data:
                for field, value in sample_result.data.items():
                    print(f"  {field}: {value}")

    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")
        raise


async def demonstrate_with_simulated_failures():
    """Demonstrate batch processing with simulated failures."""
    print("\n=== Demo with Simulated Failures ===\n")

    # Create mix of valid and invalid files
    mock_files = [
        MockFile("good_file_1.pdf", b"Valid PDF content 1"),
        MockFile("corrupted_file.pdf", b""),  # Empty file - will fail
        MockFile("good_file_2.pdf", b"Valid PDF content 2"),
        MockFile("invalid_format.txt", b"This is not a PDF"),  # Wrong format
        MockFile("good_file_3.pdf", b"Valid PDF content 3"),
    ]

    # Setup with mock components that simulate failures
    mock_repository = Mock()
    mock_repository.save_extraction.return_value = 999

    async_pdf_processor = AsyncPDFProcessor(mock_repository)
    batch_processor = AsyncBatchProcessor(
        pdf_processor=async_pdf_processor,
        max_concurrent=2,
        progress_callback=progress_handler
    )

    # Mock extractor
    extractor = Mock()
    extractor.extract.return_value = {"field1": "value1", "field2": "value2"}

    print("Processing files with simulated failures...\n")

    start_time = time.time()

    try:
        results = await batch_processor.process_batch(
            uploaded_files=mock_files,
            extractor=extractor,
            fields=["field1", "field2"],
            method="mock"
        )

        processing_time = time.time() - start_time

        # Analyze results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"\n=== Results Summary ===")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")

        if failed:
            print(f"\nFailures:")
            for result in failed:
                print(f"  - {result.file_name}: {result.error}")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")


async def compare_sync_vs_async_performance():
    """Compare synchronous vs asynchronous processing performance."""
    print("\n=== Sync vs Async Performance Comparison ===\n")

    # Create test files
    test_files = [MockFile(f"perf_test_{i}.pdf", f"Content {i}".encode()) for i in range(10)]

    print(f"Processing {len(test_files)} files...")

    # Mock components
    mock_repository = Mock()
    mock_repository.save_extraction.return_value = 123
    mock_extractor = Mock()
    mock_extractor.extract.return_value = {"field": "value"}

    # Simulate async processing time
    async def simulate_processing_delay():
        await asyncio.sleep(0.1)  # 100ms per file
        return ({"field": "value"}, "hash123")

    # Test async processing
    print("Testing async processing...")
    async_processor = AsyncPDFProcessor(mock_repository)
    batch_processor = AsyncBatchProcessor(
        async_processor,
        max_concurrent=5,  # High concurrency
        progress_callback=None
    )

    # Mock the processing to include delay
    import unittest.mock
    with unittest.mock.patch.object(async_processor, 'process_file_async', side_effect=lambda *args: simulate_processing_delay()), \
         unittest.mock.patch.object(async_processor, 'save_extraction_result_async', return_value=123):

        start_time = time.time()
        results = await batch_processor.process_batch(test_files, mock_extractor, ["field"], "test")
        async_time = time.time() - start_time

    # Calculate theoretical sync time (sequential processing)
    theoretical_sync_time = len(test_files) * 0.1

    print(f"\n=== Performance Results ===")
    print(f"Files processed: {len(test_files)}")
    print(f"Async processing time: {async_time:.2f} seconds")
    print(f"Theoretical sync time: {theoretical_sync_time:.2f} seconds")
    print(f"Speed improvement: {theoretical_sync_time/async_time:.1f}x faster")
    print(f"Efficiency: {(1 - async_time/theoretical_sync_time)*100:.1f}% time saved")


async def main():
    """Main function to run all demonstrations."""
    try:
        # Run basic demonstration
        await demonstrate_async_batch_processing()

        # Run failure handling demonstration
        await demonstrate_with_simulated_failures()

        # Run performance comparison
        await compare_sync_vs_async_performance()

        print("\n🎉 All demonstrations completed successfully!")

    except Exception as e:
        print(f"\n💥 Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async demonstrations
    asyncio.run(main())
