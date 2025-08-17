"""PDF processor for the PDF parser application.

This module contains the PDFProcessor class that orchestrates the complete
PDF processing workflow including validation, text extraction, data
extraction, and result persistence.
"""

from hashlib import sha256
from typing import Dict, List, Tuple

from ..database import ExtractionRepository
from ..extractors import DataExtractor, TextExtractor
from ..validators import PDFValidator
from ..exceptions import ValidationError, PDFProcessingError, DataExtractionError

__all__ = ["PDFProcessor"]


class PDFProcessor:
    """Main service class for PDF file processing operations.

    This class orchestrates the complete PDF processing workflow including
    validation, text extraction, data extraction, and result persistence.
    It serves as the main entry point for processing operations.

    Attributes:
        repository: Repository for data persistence operations
        text_extractor: Service for PDF text extraction
        validator: Service for PDF file validation
    """

    def __init__(self, repository: ExtractionRepository) -> None:
        """Initialize PDF processor with required dependencies.

        Args:
            repository: Repository instance for data persistence
        """
        self.repository: ExtractionRepository = repository
        self.text_extractor: TextExtractor = TextExtractor()
        self.validator: PDFValidator = PDFValidator()

    def process_file(self, file_bytes: bytes, filename: str,
                    extractor: DataExtractor, fields: List[str]) -> Tuple[Dict[str, str], str]:
        """Process a single PDF file through the complete workflow.

        Executes validation, text extraction, and data extraction
        in sequence. Returns the extracted data and file hash.

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
        self.validator.validate_pdf_file(file_bytes, filename)
        text: str = self.text_extractor.extract_text(file_bytes)
        file_hash: str = sha256(file_bytes).hexdigest()[:6]

        data: Dict[str, str] = extractor.extract(text, fields)

        return data, file_hash

    def save_extraction_result(self, filename: str, file_hash: str,
                              method: str, data: Dict[str, str]) -> int:
        """Save extraction result to persistent storage.

        Delegates to the repository for actual data persistence.

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
        return self.repository.save_extraction(filename, file_hash, method, data)
