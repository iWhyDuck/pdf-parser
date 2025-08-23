"""Asynchronous PDF processor for the PDF parser application.

This module contains the AsyncPDFProcessor class that orchestrates the complete
PDF processing workflow including validation, text extraction, data
extraction, and result persistence using asyncio for concurrent operations.
"""

import asyncio
from hashlib import sha256
from typing import Dict, List, Tuple

from ..database import ExtractionRepository
from ..extractors import DataExtractor, TextExtractor
from ..validators import PDFValidator
from ..exceptions import ValidationError, PDFProcessingError, DataExtractionError

__all__ = ["AsyncPDFProcessor"]


class AsyncPDFProcessor:
    """Asynchronous service class for PDF file processing operations.

    This class orchestrates the complete PDF processing workflow including
    validation, text extraction, data extraction, and result persistence
    using asyncio for improved performance with concurrent operations.

    Attributes:
        repository: Repository for data persistence operations
        text_extractor: Service for PDF text extraction
        validator: Service for PDF file validation
    """

    def __init__(self, repository: ExtractionRepository) -> None:
        """Initialize async PDF processor with required dependencies.

        Args:
            repository: Repository instance for data persistence
        """
        self.repository: ExtractionRepository = repository
        self.text_extractor: TextExtractor = TextExtractor()
        self.validator: PDFValidator = PDFValidator()

    async def process_file_async(
        self,
        file_bytes: bytes,
        filename: str,
        extractor: DataExtractor,
        fields: List[str]
    ) -> Tuple[Dict[str, str], str]:
        """Process a single PDF file asynchronously through the complete workflow.

        Executes validation, text extraction, and data extraction
        in sequence using thread pool for CPU-intensive operations.
        Returns the extracted data and file hash.

        Args:
            file_bytes: Raw PDF file content
            filename: Original filename for validation and error reporting
            extractor: Data extraction strategy to use
            fields: List of fields to extract

        Returns:
            Tuple containing extracted data dictionary and file hash

        Raises:
            ValidationError: If file validation fails
            PDFProcessingError: If text extraction fails
            DataExtractionError: If data extraction fails
        """
        # Run validation in thread pool to avoid blocking
        await asyncio.to_thread(self.validator.validate_pdf_file, file_bytes, filename)

        # Run text extraction in thread pool (CPU-intensive)
        text = await asyncio.to_thread(self.text_extractor.extract_text, file_bytes)

        # Calculate file hash in thread pool
        file_hash = await asyncio.to_thread(self._calculate_file_hash, file_bytes)

        # Run data extraction in thread pool (potentially CPU/AI intensive)
        data = await asyncio.to_thread(extractor.extract, text, fields)

        return data, file_hash

    async def save_extraction_result_async(
        self,
        filename: str,
        file_hash: str,
        method: str,
        data: Dict[str, str]
    ) -> int:
        """Save extraction result to persistent storage asynchronously.

        Delegates to the repository for actual data persistence using
        thread pool to avoid blocking on database I/O.

        Args:
            filename: Original PDF filename
            file_hash: SHA256 hash of file content
            method: Extraction method used ('classic' or 'ai')
            data: Extracted field-value pairs

        Returns:
            Database ID of the saved extraction record

        Raises:
            DatabaseError: If persistence operation fails
        """
        return await asyncio.to_thread(
            self.repository.save_extraction, filename, file_hash, method, data
        )

    def _calculate_file_hash(self, file_bytes: bytes) -> str:
        """Calculate SHA256 hash of file content.

        Args:
            file_bytes: Raw file content

        Returns:
            First 6 characters of SHA256 hash
        """
        return sha256(file_bytes).hexdigest()[:6]

    async def process_file_sync_wrapper(
        self,
        file_bytes: bytes,
        filename: str,
        extractor: DataExtractor,
        fields: List[str]
    ) -> Tuple[Dict[str, str], str]:
        """Synchronous wrapper for async process_file_async method.

        Provides backward compatibility for synchronous code.
        Use process_file_async directly in async contexts.

        Args:
            file_bytes: Raw PDF file content
            filename: Original filename for validation and error reporting
            extractor: Data extraction strategy to use
            fields: List of fields to extract

        Returns:
            Tuple containing extracted data dictionary and file hash
        """
        return await self.process_file_async(file_bytes, filename, extractor, fields)

    async def save_extraction_result_sync_wrapper(
        self,
        filename: str,
        file_hash: str,
        method: str,
        data: Dict[str, str]
    ) -> int:
        """Synchronous wrapper for async save_extraction_result_async method.

        Provides backward compatibility for synchronous code.
        Use save_extraction_result_async directly in async contexts.

        Args:
            filename: Original PDF filename
            file_hash: SHA256 hash of file content
            method: Extraction method used ('classic' or 'ai')
            data: Extracted field-value pairs

        Returns:
            Database ID of the saved extraction record
        """
        return await self.save_extraction_result_async(filename, file_hash, method, data)
