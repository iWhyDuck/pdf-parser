"""Batch processor for the PDF parser application.

This module contains the BatchProcessor class that handles batch processing
of multiple PDF files, providing progress feedback and error handling for
individual file processing failures.
"""

from typing import Any, Dict, List

import streamlit as st

try:
    from langfuse import observe
except ImportError:
    def observe(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func
        return decorator if args else decorator

from ..extractors import DataExtractor
from .pdf_processor import PDFProcessor

__all__ = ["BatchProcessor"]


class BatchProcessor:
    """Processes multiple PDF files in batch operations.

    This class handles batch processing of multiple PDF files,
    providing progress feedback and error handling for individual
    file processing failures.

    Attributes:
        pdf_processor: PDFProcessor instance for individual file processing
    """

    def __init__(self, pdf_processor: PDFProcessor) -> None:
        """Initialize batch processor with PDF processor dependency.

        Args:
            pdf_processor: PDFProcessor instance for file processing
        """
        self.pdf_processor: PDFProcessor = pdf_processor

    @observe(name="batch_pdf_processing")
    def process_batch(self, uploaded_files: List[Any], extractor: DataExtractor,
                     fields: List[str], method: str) -> List[Dict[str, Any]]:
        """Process multiple PDF files in batch.

        Processes each file individually and collects results.
        Continues processing even if individual files fail,
        providing partial results for successful files.

        Args:
            uploaded_files: List of uploaded file objects
            extractor: Data extraction strategy to use
            fields: List of fields to extract from each file
            method: Extraction method name for database storage

        Returns:
            List of dictionaries containing results for each file,
            including either successful extraction data or error information
        """
        batch_results: List[Dict[str, Any]] = []

        for uploaded_file in uploaded_files:
            try:
                pdf_bytes: bytes = uploaded_file.read()
                data, file_hash = self.pdf_processor.process_file(
                    pdf_bytes, uploaded_file.name, extractor, fields
                )

                db_id: int = self.pdf_processor.save_extraction_result(
                    uploaded_file.name, file_hash, method, data
                )

                batch_results.append({
                    "file": uploaded_file.name,
                    "result": data,
                    "db_id": db_id
                })
                st.write(f"✅ Processed: {uploaded_file.name}")

            except Exception as e:
                batch_results.append({
                    "file": uploaded_file.name,
                    "error": str(e)
                })
                st.write(f"❌ Error in {uploaded_file.name}: {str(e)}")

        return batch_results
