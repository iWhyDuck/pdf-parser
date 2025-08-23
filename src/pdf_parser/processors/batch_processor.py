"""Asynchronous batch processor for the PDF parser application.

This module contains the AsyncBatchProcessor class that handles concurrent batch processing
of multiple PDF files, providing progress feedback and error handling for
individual file processing failures using asyncio.
"""

import asyncio
from typing import Any, Dict, List, Callable, Optional, Protocol
from dataclasses import dataclass
from enum import Enum

try:
    from langfuse import observe
except ImportError:
    def observe(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func
        return decorator if args else decorator

from ..extractors import DataExtractor
from .async_pdf_processor import AsyncPDFProcessor

__all__ = ["AsyncBatchProcessor", "BatchResult", "ProgressEvent", "ProgressCallback"]


class ProgressEventType(Enum):
    """Types of progress events."""
    BATCH_STARTED = "batch_started"
    FILE_STARTED = "file_started"
    FILE_COMPLETED = "file_completed"
    FILE_FAILED = "file_failed"
    BATCH_COMPLETED = "batch_completed"


@dataclass
class ProgressEvent:
    """Progress event data structure."""
    event_type: ProgressEventType
    file_name: Optional[str] = None
    current_file: int = 0
    total_files: int = 0
    message: str = ""
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Result of batch processing operation."""
    file_name: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    db_id: Optional[int] = None
    error: Optional[str] = None


class ProgressCallback(Protocol):
    """Protocol for progress callback functions."""
    def __call__(self, event: ProgressEvent) -> None:
        """Handle progress event."""
        ...


class AsyncBatchProcessor:
    """Asynchronously processes multiple PDF files in batch operations.

    This class handles concurrent batch processing of multiple PDF files,
    providing progress feedback and error handling for individual
    file processing failures using asyncio.

    Attributes:
        pdf_processor: AsyncPDFProcessor instance for individual file processing
        max_concurrent: Maximum number of concurrent file processing operations
        progress_callback: Optional callback function for progress updates
    """

    def __init__(
        self,
        pdf_processor: AsyncPDFProcessor,
        max_concurrent: int = 5,
        progress_callback: Optional[ProgressCallback] = None
    ) -> None:
        """Initialize async batch processor.

        Args:
            pdf_processor: AsyncPDFProcessor instance for file processing
            max_concurrent: Maximum number of concurrent operations (default: 5)
            progress_callback: Optional callback for progress updates
        """
        self.pdf_processor: AsyncPDFProcessor = pdf_processor
        self.max_concurrent: int = max_concurrent
        self.progress_callback: Optional[ProgressCallback] = progress_callback
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent)

    def _emit_progress(self, event: ProgressEvent) -> None:
        """Emit progress event to callback if available."""
        if self.progress_callback:
            self.progress_callback(event)

    async def _process_single_file(
        self,
        uploaded_file: Any,
        extractor: DataExtractor,
        fields: List[str],
        method: str,
        file_index: int,
        total_files: int
    ) -> BatchResult:
        """Process a single PDF file asynchronously.

        Args:
            uploaded_file: Uploaded file object
            extractor: Data extraction strategy to use
            fields: List of fields to extract from the file
            method: Extraction method name for database storage
            file_index: Current file index for progress tracking
            total_files: Total number of files being processed

        Returns:
            BatchResult containing either successful extraction data or error information
        """
        async with self._semaphore:
            file_name = uploaded_file.name

            self._emit_progress(ProgressEvent(
                event_type=ProgressEventType.FILE_STARTED,
                file_name=file_name,
                current_file=file_index + 1,
                total_files=total_files,
                message=f"Starting processing of {file_name}"
            ))

            try:
                # Read file content asynchronously
                pdf_bytes = await self._read_file_async(uploaded_file)

                # Process file asynchronously
                data, file_hash = await self.pdf_processor.process_file_async(
                    pdf_bytes, file_name, extractor, fields
                )

                # Save to database asynchronously
                db_id = await self.pdf_processor.save_extraction_result_async(
                    file_name, file_hash, method, data
                )

                result = BatchResult(
                    file_name=file_name,
                    success=True,
                    data=data,
                    db_id=db_id
                )

                self._emit_progress(ProgressEvent(
                    event_type=ProgressEventType.FILE_COMPLETED,
                    file_name=file_name,
                    current_file=file_index + 1,
                    total_files=total_files,
                    message=f"Successfully processed {file_name}"
                ))

                return result

            except Exception as e:
                error_msg = str(e)
                result = BatchResult(
                    file_name=file_name,
                    success=False,
                    error=error_msg
                )

                self._emit_progress(ProgressEvent(
                    event_type=ProgressEventType.FILE_FAILED,
                    file_name=file_name,
                    current_file=file_index + 1,
                    total_files=total_files,
                    message=f"Failed to process {file_name}",
                    error=error_msg
                ))

                return result

    async def _read_file_async(self, uploaded_file: Any) -> bytes:
        """Read file content asynchronously.

        Note: This assumes uploaded_file has async read capability.
        If not, wrap in asyncio.to_thread() for I/O operations.
        """
        # If uploaded_file doesn't support async, use thread pool
        if hasattr(uploaded_file, 'aread') and callable(getattr(uploaded_file, 'aread')):
            # Check if aread is actually async
            aread_method = getattr(uploaded_file, 'aread')
            if asyncio.iscoroutinefunction(aread_method):
                return await uploaded_file.aread()
            else:
                # It's not actually async, fall back to thread pool
                return await asyncio.to_thread(uploaded_file.read)
        else:
            # Fallback to thread pool for sync I/O
            return await asyncio.to_thread(uploaded_file.read)

    @observe(name="async_batch_pdf_processing")
    async def process_batch(
        self,
        uploaded_files: List[Any],
        extractor: DataExtractor,
        fields: List[str],
        method: str
    ) -> List[BatchResult]:
        """Process multiple PDF files concurrently in batch.

        Processes files concurrently with controlled parallelism.
        Continues processing even if individual files fail,
        providing partial results for successful files.

        Args:
            uploaded_files: List of uploaded file objects
            extractor: Data extraction strategy to use
            fields: List of fields to extract from each file
            method: Extraction method name for database storage

        Returns:
            List of BatchResult objects containing results for each file,
            including either successful extraction data or error information
        """
        total_files = len(uploaded_files)

        self._emit_progress(ProgressEvent(
            event_type=ProgressEventType.BATCH_STARTED,
            total_files=total_files,
            message=f"Starting batch processing of {total_files} files"
        ))

        # Create tasks for concurrent processing
        tasks = [
            self._process_single_file(
                uploaded_file, extractor, fields, method, index, total_files
            )
            for index, uploaded_file in enumerate(uploaded_files)
        ]

        # Process all files concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that weren't caught in individual tasks
        final_results = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                final_results.append(BatchResult(
                    file_name=uploaded_files[i].name,
                    success=False,
                    error=str(result)
                ))
            else:
                final_results.append(result)

        successful_count = sum(1 for r in final_results if r.success)
        failed_count = sum(1 for r in final_results if not r.success)

        self._emit_progress(ProgressEvent(
            event_type=ProgressEventType.BATCH_COMPLETED,
            total_files=total_files,
            message=f"Batch processing completed. {successful_count} successful, {failed_count} failed"
        ))

        return final_results

    async def process_batch_with_progress(
        self,
        uploaded_files: List[Any],
        extractor: DataExtractor,
        fields: List[str],
        method: str
    ) -> List[BatchResult]:
        """Process batch with real-time progress updates using asyncio.wait.

        Alternative method that provides more granular progress updates.
        """
        total_files = len(uploaded_files)

        self._emit_progress(ProgressEvent(
            event_type=ProgressEventType.BATCH_STARTED,
            total_files=total_files,
            message=f"Starting batch processing of {total_files} files"
        ))

        # Create tasks with metadata mapping
        task_to_info = {}
        tasks = []

        for index, uploaded_file in enumerate(uploaded_files):
            task = asyncio.create_task(
                self._process_single_file(
                    uploaded_file, extractor, fields, method, index, total_files
                )
            )
            tasks.append(task)
            task_to_info[task] = (index, uploaded_file.name)

        results = [None] * total_files

        # Wait for all tasks to complete
        done_tasks, _ = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        # Process results
        for task in done_tasks:
            index, file_name = task_to_info[task]
            try:
                if task.exception():
                    results[index] = BatchResult(
                        file_name=file_name,
                        success=False,
                        error=str(task.exception())
                    )
                else:
                    result = task.result()
                    results[index] = result
            except Exception as e:
                results[index] = BatchResult(
                    file_name=file_name,
                    success=False,
                    error=str(e)
                )

        successful_count = sum(1 for r in results if r and r.success)
        failed_count = sum(1 for r in results if r and not r.success)

        self._emit_progress(ProgressEvent(
            event_type=ProgressEventType.BATCH_COMPLETED,
            total_files=total_files,
            message=f"Batch processing completed. {successful_count} successful, {failed_count} failed"
        ))

        return [r for r in results if r is not None]
