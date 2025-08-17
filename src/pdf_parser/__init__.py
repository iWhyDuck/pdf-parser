"""PDF Parser - A comprehensive PDF data extraction toolkit.

This package provides tools for extracting structured data from PDF documents
using both traditional regex-based methods and modern AI-powered approaches.

The package is organized into the following modules:
- config: Application configuration and settings
- exceptions: Custom exception classes
- models: Database models and data structures
- database: Database management and repositories
- validators: PDF file validation utilities
- extractors: Text and data extraction implementations
- processors: High-level processing workflows
- ui: Streamlit user interface components
"""

__version__ = "1.0.0"
__author__ = "PDF Parser Team"
__description__ = "PDF data extraction toolkit with regex and AI capabilities"

from .config import Config
from .exceptions import (
    PDFProcessingError,
    DataExtractionError,
    DatabaseError,
    ValidationError
)
from .models import Base, Extraction
from .database import DatabaseManager, ExtractionRepository
from .validators import PDFValidator
from .extractors import TextExtractor, DataExtractor, ClassicExtractor, AIExtractor
from .processors import PDFProcessor, BatchProcessor
from .ui import FieldSelector, UIRenderer

__all__ = [
    # Configuration
    "Config",
    # Exceptions
    "PDFProcessingError",
    "DataExtractionError",
    "DatabaseError",
    "ValidationError",
    # Models
    "Base",
    "Extraction",
    # Database
    "DatabaseManager",
    "ExtractionRepository",
    # Validators
    "PDFValidator",
    # Extractors
    "TextExtractor",
    "DataExtractor",
    "ClassicExtractor",
    "AIExtractor",
    # Processors
    "PDFProcessor",
    "BatchProcessor",
    # UI
    "FieldSelector",
    "UIRenderer"
]
