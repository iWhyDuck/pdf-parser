"""Simple demonstration script for AsyncBatchProcessor.

This script shows basic usage of the asynchronous batch processor
with mock data to demonstrate concurrent processing capabilities.
"""

import asyncio
import time
from typing import List
from unittest.mock import Mock

from src.pdf_parser.processors import (
    AsyncPDFProcessor,
    AsyncBatchProcessor,
    ProgressEvent,
    ProgressEventType
)
from src.pdf_parser.database import DatabaseManager, ExtractionRepository


class MockFile:
    """Mock file object for demonstration."""

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def read(self) -> bytes:
        """Read file content."""
        return self._content


class MockExtractor:
    """Mock data extractor for demonstration."""

    def extract(self, text: str, fields: List[str]) -> dict:
        """Extract mock data from text."""
        return {field: f"extracted_{field}_value" for field in fields}


def simple_progress_callback(event: ProgressEvent) -> None:
    """Simple progress callback that prints events."""
    timestamp = time.strftime("%H:%M:%S")

    if event.event_type == ProgressEventType.BATCH_STARTED:
        print(f"🚀 [{timestamp}] Starting batch processing of {event.total_files} files")

    elif event.event_type == ProgressEventType.FILE_STARTED:
        print(f"📄 [{timestamp}] Processing {event.file_name} ({event.current_file}/{event.total_files})")

    elif event.event_type == ProgressEventType.FILE_COMPLETED:
        print(f"✅ [{timestamp}] Completed {event.file_name}")

    elif event.event_type == ProgressEventType.FILE_FAILED:
        print(f"❌ [{timestamp}] Failed {event.file_name}: {event.error}")

    elif event.event_type == ProgressEventType.BATCH_COMPLETED:
        print(f"🎉 [{timestamp}] {event.message}")


async def demonstrate_async_batch_processing():
    """Demonstrate async batch processing with mock components."""
    print("=== AsyncBatchProcessor Demonstration ===\n")

    # Create mock files
    mock_files = [
        MockFile("document_1.pdf", b"Mock PDF content for document 1"),
        MockFile("document_2.pdf", b"Mock PDF content for document 2"),
        MockFile("document_3.pdf", b"Mock PDF content for document 3"),
        MockFile("document_4.pdf", b"Mock PDF content for document 4"),
        MockFile("document_5.pdf", b"Mock PDF content for document 5"),
    ]

    # Create mock repository
    mock_repo = Mock()
    mock_repo.save_extraction.return_value = 12345

    # Create processors
    async_pdf_processor = AsyncPDFProcessor(mock_repo)
    batch_processor = AsyncBatchProcessor(
        pdf_processor=async_pdf_processor,
        max_concurrent=3,  # Process max 3 files at once
        progress_callback=simple_progress_callback
    )

    # Create mock extractor
    mock_extractor = MockExtractor()
    fields = ["invoice_number", "date", "amount"]

    print(f"Configuration:")
    print(f"  - Files to process: {len(mock_files)}")
    print(f"  - Max concurrent: 3")
    print(f"  - Fields to extract: {', '.join(fields)}")
    print()

    # Mock the validation, text extraction and hash calculation
    # to simulate successful processing
    async_pdf_processor.validator.validate_pdf_file = Mock()
    async_pdf_processor.text_extractor.extract_text = Mock(return_value="Mock extracted text")
    async_pdf_processor._calculate_file_hash = Mock(return_value="abc123")

    start_time = time.time()

    try:
        # Process files asynchronously
        results = await batch_processor.process_batch(
            uploaded_files=mock_files,
            extractor=mock_extractor,
            fields=fields,
            method="mock"
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Display results
        print(f"\n=== Results ===")
        print(f"Processing time: {processing_time:.3f} seconds")
        print(f"Files processed: {len(results)}")

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")

        if successful:
            print(f"\nSample extracted data from {successful[0].file_name}:")
            for field, value in successful[0].data.items():
                print(f"  {field}: {value}")

        if failed:
            print(f"\nFailures:")
            for result in failed:
                print(f"  - {result.file_name}: {result.error}")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        raise


async def demonstrate_concurrency_benefits():
    """Demonstrate the benefits of concurrent processing."""
    print("\n=== Concurrency Benefits Demo ===\n")

    # Create larger set of files
    mock_files = [
        MockFile(f"doc_{i:03d}.pdf", f"Content {i}".encode())
        for i in range(1, 11)
    ]

    # Mock repository
    mock_repo = Mock()
    mock_repo.save_extraction.return_value = 999

    print(f"Processing {len(mock_files)} files...")

    # Test with different concurrency levels
    concurrency_levels = [1, 3, 5]

    for max_concurrent in concurrency_levels:
        print(f"\n--- Max concurrent: {max_concurrent} ---")

        # Create processor
        async_pdf_processor = AsyncPDFProcessor(mock_repo)
        batch_processor = AsyncBatchProcessor(
            pdf_processor=async_pdf_processor,
            max_concurrent=max_concurrent,
            progress_callback=None  # No progress for cleaner output
        )

        # Mock dependencies with small delay to simulate work
        async_pdf_processor.validator.validate_pdf_file = Mock()
        async_pdf_processor.text_extractor.extract_text = Mock(return_value="Mock text")
        async_pdf_processor._calculate_file_hash = Mock(return_value="hash123")

        # Add artificial delay to simulate real processing
        original_process = async_pdf_processor.process_file_async

        async def delayed_process(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms delay per file
            return await original_process(*args, **kwargs)

        async_pdf_processor.process_file_async = delayed_process

        start_time = time.time()

        results = await batch_processor.process_batch(
            uploaded_files=mock_files,
            extractor=MockExtractor(),
            fields=["field1", "field2"],
            method="demo"
        )

        end_time = time.time()
        processing_time = end_time - start_time

        successful = sum(1 for r in results if r.success)
        theoretical_sequential_time = len(mock_files) * 0.1
        speedup = theoretical_sequential_time / processing_time

        print(f"  Time: {processing_time:.2f}s")
        print(f"  Successful: {successful}/{len(results)}")
        print(f"  Speedup: {speedup:.1f}x vs sequential")


async def main():
    """Run all demonstrations."""
    try:
        await demonstrate_async_batch_processing()
        await demonstrate_concurrency_benefits()
        print("\n🎉 All demonstrations completed successfully!")

    except Exception as e:
        print(f"\n💥 Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
