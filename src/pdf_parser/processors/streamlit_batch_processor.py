"""Streamlit-compatible wrapper for AsyncBatchProcessor.

This module provides a synchronous wrapper around the AsyncBatchProcessor
that can be used in Streamlit applications. It handles the async-to-sync
conversion and provides Streamlit-specific progress reporting.
"""

import asyncio
from typing import Any, Dict, List, Optional
import streamlit as st

try:
    from langfuse import observe
except ImportError:
    def observe(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func
        return decorator if args else decorator

from .async_pdf_processor import AsyncPDFProcessor
from .batch_processor import AsyncBatchProcessor, ProgressEvent, ProgressEventType, BatchResult
from ..extractors import DataExtractor

__all__ = ["StreamlitBatchProcessor"]


class StreamlitBatchProcessor:
    """Streamlit-compatible wrapper for AsyncBatchProcessor.

    This class provides a synchronous interface to the AsyncBatchProcessor
    while maintaining compatibility with Streamlit's execution model.
    It handles async-to-sync conversion and provides Streamlit-specific
    progress reporting using st.write().

    Attributes:
        async_pdf_processor: AsyncPDFProcessor instance for file processing
        max_concurrent: Maximum number of concurrent operations
    """

    def __init__(self, async_pdf_processor: AsyncPDFProcessor, max_concurrent: int = 5) -> None:
        """Initialize Streamlit batch processor.

        Args:
            async_pdf_processor: AsyncPDFProcessor instance for file processing
            max_concurrent: Maximum number of concurrent operations (default: 5)
        """
        self.async_pdf_processor = async_pdf_processor
        self.max_concurrent = max_concurrent
        self._progress_container: Optional[Any] = None

    def _streamlit_progress_callback(self, event: ProgressEvent) -> None:
        """Handle progress events by writing to Streamlit interface.

        Args:
            event: ProgressEvent containing progress information
        """
        if event.event_type == ProgressEventType.BATCH_STARTED:
            st.write(f"🚀 Starting batch processing of {event.total_files} files")

        elif event.event_type == ProgressEventType.FILE_STARTED:
            st.write(f"📄 Processing {event.file_name} ({event.current_file}/{event.total_files})")

        elif event.event_type == ProgressEventType.FILE_COMPLETED:
            st.write(f"✅ Processed: {event.file_name}")

        elif event.event_type == ProgressEventType.FILE_FAILED:
            st.write(f"❌ Error in {event.file_name}: {event.error}")

        elif event.event_type == ProgressEventType.BATCH_COMPLETED:
            st.write(f"🎉 {event.message}")

    @observe(name="streamlit_batch_pdf_processing")
    def process_batch(self, uploaded_files: List[Any], extractor: DataExtractor,
                     fields: List[str], method: str) -> List[Dict[str, Any]]:
        """Process multiple PDF files in batch (Streamlit-compatible).

        This method wraps the async batch processing functionality to provide
        a synchronous interface compatible with Streamlit. It handles the
        async-to-sync conversion internally.

        Args:
            uploaded_files: List of uploaded file objects
            extractor: Data extraction strategy to use
            fields: List of fields to extract from each file
            method: Extraction method name for database storage

        Returns:
            List of dictionaries containing results for each file,
            including either successful extraction data or error information
        """
        # Create the async batch processor with Streamlit progress callback
        async_batch_processor = AsyncBatchProcessor(
            pdf_processor=self.async_pdf_processor,
            max_concurrent=self.max_concurrent,
            progress_callback=self._streamlit_progress_callback
        )

        # Run the async batch processing
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're already in an event loop, we need to run in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        async_batch_processor.process_batch(uploaded_files, extractor, fields, method)
                    )
                    batch_results = future.result()
            except RuntimeError:
                # No running loop, we can use asyncio.run directly
                batch_results = asyncio.run(
                    async_batch_processor.process_batch(uploaded_files, extractor, fields, method)
                )

        except Exception as e:
            st.error(f"❌ Batch processing failed: {str(e)}")
            return []

        # Convert BatchResult objects to dictionary format for backward compatibility
        legacy_results = []
        for result in batch_results:
            if result.success:
                legacy_results.append({
                    "file": result.file_name,
                    "result": result.data,
                    "db_id": result.db_id
                })
            else:
                legacy_results.append({
                    "file": result.file_name,
                    "error": result.error
                })

        return legacy_results

    def process_batch_with_progress_bar(self, uploaded_files: List[Any], extractor: DataExtractor,
                                      fields: List[str], method: str) -> List[Dict[str, Any]]:
        """Process batch with Streamlit progress bar.

        Alternative method that uses Streamlit's progress bar component
        for visual progress tracking.

        Args:
            uploaded_files: List of uploaded file objects
            extractor: Data extraction strategy to use
            fields: List of fields to extract from each file
            method: Extraction method name for database storage

        Returns:
            List of dictionaries containing results for each file
        """
        total_files = len(uploaded_files)
        progress_bar = st.progress(0)
        status_text = st.empty()

        completed_files = 0
        results = []

        def progress_callback_with_bar(event: ProgressEvent) -> None:
            nonlocal completed_files

            if event.event_type == ProgressEventType.BATCH_STARTED:
                status_text.text(f"Starting batch processing of {total_files} files...")

            elif event.event_type == ProgressEventType.FILE_STARTED:
                status_text.text(f"Processing {event.file_name}...")

            elif event.event_type in [ProgressEventType.FILE_COMPLETED, ProgressEventType.FILE_FAILED]:
                completed_files += 1
                progress = completed_files / total_files
                progress_bar.progress(progress)

                if event.event_type == ProgressEventType.FILE_COMPLETED:
                    status_text.text(f"✅ Completed {event.file_name} ({completed_files}/{total_files})")
                else:
                    status_text.text(f"❌ Failed {event.file_name} ({completed_files}/{total_files})")

            elif event.event_type == ProgressEventType.BATCH_COMPLETED:
                progress_bar.progress(1.0)
                status_text.text("🎉 Batch processing completed!")

        # Create async batch processor with progress bar callback
        async_batch_processor = AsyncBatchProcessor(
            pdf_processor=self.async_pdf_processor,
            max_concurrent=self.max_concurrent,
            progress_callback=progress_callback_with_bar
        )

        try:
            # Run async processing
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        async_batch_processor.process_batch(uploaded_files, extractor, fields, method)
                    )
                    batch_results = future.result()
            except RuntimeError:
                batch_results = asyncio.run(
                    async_batch_processor.process_batch(uploaded_files, extractor, fields, method)
                )

        except Exception as e:
            status_text.text(f"❌ Batch processing failed: {str(e)}")
            progress_bar.empty()
            return []

        # Convert to legacy format
        legacy_results = []
        for result in batch_results:
            if result.success:
                legacy_results.append({
                    "file": result.file_name,
                    "result": result.data,
                    "db_id": result.db_id
                })
            else:
                legacy_results.append({
                    "file": result.file_name,
                    "error": result.error
                })

        return legacy_results

    def get_async_processor(self) -> AsyncBatchProcessor:
        """Get the underlying AsyncBatchProcessor for advanced usage.

        Returns:
            AsyncBatchProcessor instance
        """
        return AsyncBatchProcessor(
            pdf_processor=self.async_pdf_processor,
            max_concurrent=self.max_concurrent
        )
