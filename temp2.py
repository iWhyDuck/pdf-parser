# app.py – Streamlit demo with two PDF data extraction modes:
# 1) Classic (Regex)   2) AI (GPT-3.5 quick-scan → detailed)
# Dependencies: streamlit, pdfplumber, openai, sqlalchemy, concurrent.futures, langfuse

from __future__ import annotations
import io, json, os, re
from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Protocol

from dotenv import load_dotenv
load_dotenv()

import pdfplumber
import streamlit as st
from openai import OpenAIError

# FIXED: Langfuse imports for v3 with fallback support
try:
    from langfuse.openai import OpenAI
    from langfuse import observe
except ImportError:
    # Fallback for older SDK version (v2)
    try:
        from langfuse.openai import OpenAI
        from langfuse.decorators import observe
    except ImportError:
        st.error("❌ Cannot import Langfuse. Install with: pip install langfuse --upgrade")
        # Mock decorator as fallback
        def observe(*args: Any, **kwargs: Any) -> Any:
            def decorator(func: Any) -> Any:
                return func
            return decorator if args else decorator
        # Mock OpenAI client
        from openai import OpenAI

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

import concurrent.futures

# ──────────────────────── 0. Custom Exceptions ────────────────────────

class PDFProcessingError(Exception):
    """Exception raised during PDF file processing operations.
    
    This exception is raised when there are issues with PDF file reading,
    parsing, or text extraction.
    """
    pass

class DataExtractionError(Exception):
    """Exception raised during data extraction operations.
    
    This exception is raised when there are issues with regex matching,
    AI processing, or field value extraction.
    """
    pass

class DatabaseError(Exception):
    """Exception raised during database operations.
    
    This exception is raised when there are issues with database
    connectivity, queries, or data persistence.
    """
    pass

class ValidationError(Exception):
    """Exception raised during data validation.
    
    This exception is raised when input data fails validation checks
    such as file format, size, or content validation.
    """
    pass

# ──────────────────────── 1. Configuration ────────────────────────

class Config:
    """Configuration class containing application settings and constants.
    
    This class centralizes all configuration parameters including
    model settings, file size limits, database configuration,
    and regex field definitions.
    """
    
    # OpenAI model configuration
    OPENAI_MODEL: str = "gpt-3.5-turbo-1106"
    
    # File validation limits
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB maximum file size
    MIN_FILE_SIZE: int = 100  # 100 bytes minimum file size
    
    # Database configuration
    DATABASE_URL: str = "sqlite:///extractions.db"
    
    # Regex field definitions for classic extraction
    REGEX_FIELDS: Dict[str, Dict[str, Union[str, List[str]]]] = {
        "customer_name": {
            "display": "Customer Name",
            "patterns": [r"Customer Name[:\s]*([A-Za-zÀ-ž ,.'-]+)"],
        },
        "policy_number": {
            "display": "Policy Number",
            "patterns": [r"Policy Number[:\s]*([\w-]+)"],
        },
        "claim_amount": {
            "display": "Claim Amount",
            "patterns": [r"Claim Amount[:\s]*\$?([\d,]+\.\d{2})"],
        },
    }

# ──────────────────────── 2. Database Models & Repository ────────────────────────

Base = declarative_base()

class Extraction(Base):
    """SQLAlchemy model for storing PDF extraction results.
    
    This model represents a single PDF extraction operation result,
    including metadata about the file, extraction method used,
    and the extracted data in JSON format.
    
    Attributes:
        id: Primary key for the extraction record
        filename: Original name of the processed PDF file
        file_hash: SHA256 hash of the file content (first 6 characters)
        extraction_method: Method used for extraction ('classic' or 'ai')
        extracted_data: JSON string containing the extracted field-value pairs
        created_at: Timestamp of when the extraction was performed
    """
    __tablename__ = "extractions"
    
    id: int = Column(Integer, primary_key=True)
    filename: str = Column(String(255), nullable=False)
    file_hash: str = Column(String(64), nullable=False)
    extraction_method: str = Column(String(50), nullable=False)
    extracted_data: str = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class DatabaseManager:
    """Manages database connections and session creation.
    
    This class handles SQLAlchemy engine creation, database initialization,
    and provides methods for creating database sessions. It implements
    lazy initialization for better resource management.
    
    Attributes:
        database_url: SQLAlchemy database URL
        _engine: Cached SQLAlchemy engine instance
        _session_factory: Cached sessionmaker factory
    """
    
    def __init__(self, database_url: str = Config.DATABASE_URL) -> None:
        """Initialize DatabaseManager with database URL.
        
        Args:
            database_url: SQLAlchemy database URL string
        """
        self.database_url: str = database_url
        self._engine: Optional[Any] = None
        self._session_factory: Optional[Any] = None
    
    @property
    def engine(self) -> Any:
        """Get or create SQLAlchemy engine with lazy initialization.
        
        Creates the engine with SQLite-specific configuration including
        connection timeout and static connection pooling.
        
        Returns:
            SQLAlchemy engine instance
            
        Raises:
            DatabaseError: If engine creation or database initialization fails
        """
        if self._engine is None:
            try:
                self._engine = create_engine(
                    self.database_url,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 20
                    },
                    poolclass=StaticPool,
                    echo=False
                )
                Base.metadata.create_all(self._engine)
            except Exception as e:
                raise DatabaseError(f"Database initialization error: {str(e)}")
        return self._engine
    
    def create_session(self) -> Session:
        """Create a new database session.
        
        Creates a new SQLAlchemy session using the configured engine.
        Each call returns a fresh session instance.
        
        Returns:
            SQLAlchemy Session instance
        """
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory()

class ExtractionRepository:
    """Repository pattern implementation for extraction data persistence.
    
    This class encapsulates all database operations related to extraction
    records, providing a clean interface for data persistence operations.
    
    Attributes:
        db_manager: DatabaseManager instance for database operations
    """
    
    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize repository with database manager.
        
        Args:
            db_manager: DatabaseManager instance for database operations
        """
        self.db_manager: DatabaseManager = db_manager
    
    def save_extraction(self, filename: str, file_hash: str, 
                       method: str, data: Dict[str, Any]) -> int:
        """Save extraction result to database.
        
        Persists the extraction result as a new record in the database.
        Handles session management and error recovery automatically.
        
        Args:
            filename: Original PDF filename
            file_hash: SHA256 hash of the file content
            method: Extraction method used ('classic' or 'ai')
            data: Dictionary containing extracted field-value pairs
            
        Returns:
            Database ID of the created extraction record
            
        Raises:
            DatabaseError: If database operation fails
        """
        session: Session = self.db_manager.create_session()
        try:
            record = Extraction(
                filename=filename,
                file_hash=file_hash,
                extraction_method=method,
                extracted_data=json.dumps(data, ensure_ascii=False)
            )
            session.add(record)
            session.commit()
            return record.id
        except SQLAlchemyError as e:
            session.rollback()
            raise DatabaseError(f"Database save error: {str(e)}")
        finally:
            session.close()

# ──────────────────────── 3. Validation ────────────────────────

class PDFValidator:
    """Validates PDF files for security and correctness.
    
    This class provides static methods for comprehensive PDF file validation
    including file size checks, format validation, and extension verification.
    All validation methods raise appropriate exceptions for different failure modes.
    """
    
    @staticmethod
    def validate_pdf_file(pdf_bytes: bytes, filename: str) -> None:
        """Perform comprehensive PDF file validation.
        
        Executes all validation checks in sequence. If any validation
        fails, appropriate exception is raised with detailed error message.
        
        Args:
            pdf_bytes: Raw PDF file content as bytes
            filename: Original filename for error reporting
            
        Raises:
            ValidationError: If any validation check fails
        """
        PDFValidator._validate_file_size(pdf_bytes, filename)
        PDFValidator._validate_pdf_format(pdf_bytes, filename)
        PDFValidator._validate_file_extension(filename)
    
    @staticmethod
    def _validate_file_size(pdf_bytes: bytes, filename: str) -> None:
        """Validate PDF file size within acceptable limits.
        
        Checks that file size is between minimum and maximum allowed values
        to prevent processing of corrupted or excessively large files.
        
        Args:
            pdf_bytes: Raw PDF file content as bytes
            filename: Original filename for error reporting
            
        Raises:
            ValidationError: If file size is outside acceptable range
        """
        if len(pdf_bytes) > Config.MAX_FILE_SIZE:
            raise ValidationError(
                f"File {filename} is too large. Maximum size: {Config.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if len(pdf_bytes) < Config.MIN_FILE_SIZE:
            raise ValidationError(f"File {filename} is too small or corrupted")
    
    @staticmethod
    def _validate_pdf_format(pdf_bytes: bytes, filename: str) -> None:
        """Validate that file content is a valid PDF format.
        
        Checks the PDF magic number at the beginning of the file
        to ensure it's a valid PDF document.
        
        Args:
            pdf_bytes: Raw PDF file content as bytes
            filename: Original filename for error reporting
            
        Raises:
            ValidationError: If file is not a valid PDF format
        """
        if not pdf_bytes.startswith(b'%PDF'):
            raise ValidationError(f"File {filename} is not a valid PDF file")
    
    @staticmethod
    def _validate_file_extension(filename: str) -> None:
        """Validate that filename has correct PDF extension.
        
        Ensures the uploaded file has the expected .pdf extension
        to prevent processing of incorrectly named files.
        
        Args:
            filename: Original filename to validate
            
        Raises:
            ValidationError: If file extension is not .pdf
        """
        if not filename.lower().endswith('.pdf'):
            raise ValidationError(
                f"Invalid file extension. Expected .pdf, got: {Path(filename).suffix}"
            )

# ──────────────────────── 4. Text Extraction ────────────────────────

class TextExtractor:
    """Extracts text content from PDF files.
    
    This class provides static methods for extracting text from PDF files
    using the pdfplumber library. Handles multi-page documents and provides
    error recovery for individual page processing failures.
    """
    
    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        """Extract text content from PDF file.
        
        Processes all pages in the PDF document and extracts text content.
        Continues processing even if individual pages fail, providing
        partial results when possible.
        
        Args:
            pdf_bytes: Raw PDF file content as bytes
            
        Returns:
            Extracted text content as a single string with page breaks
            
        Raises:
            PDFProcessingError: If PDF cannot be opened or contains no text
        """
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                if not pdf.pages:
                    raise PDFProcessingError("PDF contains no pages")
                
                text_parts: List[str] = []
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text: Optional[str] = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        st.warning(f"⚠️ Failed to process page {i+1}: {str(e)}")
                        continue
                
                if not text_parts:
                    raise PDFProcessingError("Failed to extract text from any page")
                
                return "\n".join(text_parts)
                
        except Exception as e:
            if isinstance(e, PDFProcessingError):
                raise
            raise PDFProcessingError(f"PDF reading error: {str(e)}")

# ──────────────────────── 5. Data Extractors ────────────────────────

class DataExtractor(ABC):
    """Abstract base class for data extraction implementations.
    
    This abstract class defines the interface that all data extractors
    must implement. It follows the Strategy pattern to allow different
    extraction methods to be used interchangeably.
    """
    
    @abstractmethod
    def extract(self, text: str, fields: List[str]) -> Dict[str, str]:
        """Extract data fields from text content.
        
        Abstract method that must be implemented by concrete extractors
        to perform field extraction from the provided text.
        
        Args:
            text: Input text content to extract data from
            fields: List of field names to extract
            
        Returns:
            Dictionary mapping field names to extracted values
            
        Raises:
            DataExtractionError: If extraction process fails
        """
        pass

class ClassicExtractor(DataExtractor):
    """Regex-based data extractor for structured documents.
    
    This extractor uses predefined regular expression patterns to locate
    and extract specific data fields from text content. It's optimized
    for documents with consistent formatting and known field patterns.
    
    Attributes:
        cfg: Compiled regex patterns organized by field name
    """
    
    def __init__(self, config: Dict[str, Dict[str, Union[str, List[str]]]] = Config.REGEX_FIELDS) -> None:
        """Initialize extractor with regex configuration.
        
        Compiles all regex patterns for efficient repeated use.
        Patterns are compiled with case-insensitive flag.
        
        Args:
            config: Dictionary containing field definitions and regex patterns
            
        Raises:
            DataExtractionError: If regex compilation fails
        """
        try:
            self.cfg: Dict[str, List[Any]] = {
                k: [re.compile(p, re.I) for p in v["patterns"]] 
                for k, v in config.items()
            }
        except re.error as e:
            raise DataExtractionError(f"Regex compilation error: {str(e)}")

    @observe(name="classic_extraction")
    def extract(self, text: str, fields: Optional[List[str]] = None) -> Dict[str, str]:
        """Extract data using regex patterns.
        
        Applies regex patterns to extract specified fields from text.
        If no fields are specified, attempts to extract all configured fields.
        
        Args:
            text: Input text content to extract data from
            fields: Optional list of specific fields to extract
            
        Returns:
            Dictionary mapping field names to extracted values
            
        Raises:
            DataExtractionError: If text is empty or extraction fails
        """
        if not text or not text.strip():
            raise DataExtractionError("No text content to process")
        
        try:
            out: Dict[str, str] = {}
            fields_to_extract: List[str] = fields if fields else list(self.cfg.keys())
            
            for key in fields_to_extract:
                if key not in self.cfg:
                    st.warning(f"⚠️ Unknown field: {key}")
                    continue
                
                value: Optional[str] = self._extract_field_value(key, text)
                if value:
                    out[key] = value
            
            return out
            
        except Exception as e:
            raise DataExtractionError(f"Classic extraction error: {str(e)}")
    
    def _extract_field_value(self, key: str, text: str) -> Optional[str]:
        """Extract value for a specific field using regex patterns.
        
        Tries each regex pattern for the field until a match is found.
        Returns the first successful match or None if no patterns match.
        
        Args:
            key: Field name to extract
            text: Text content to search in
            
        Returns:
            Extracted field value or None if not found
        """
        for pat in self.cfg[key]:
            try:
                match = pat.search(text)
                if match:
                    return match.group(1).strip()
            except Exception as e:
                st.warning(f"⚠️ Error processing field '{key}': {str(e)}")
                continue
        return None

class AIExtractor(DataExtractor):
    """AI-powered data extractor using OpenAI GPT models.
    
    This extractor uses OpenAI's language models to identify and extract
    data fields from unstructured text. It can discover field labels
    dynamically and extract values using natural language understanding.
    
    Attributes:
        cli: OpenAI client instance for API communication
    """
    
    def __init__(self, api_key: str) -> None:
        """Initialize AI extractor with OpenAI API key.
        
        Creates OpenAI client instance for subsequent API calls.
        
        Args:
            api_key: OpenAI API key for authentication
            
        Raises:
            DataExtractionError: If API key is missing or client creation fails
        """
        if not api_key:
            raise DataExtractionError("Missing OpenAI API key")
        
        try:
            self.cli: OpenAI = OpenAI(api_key=api_key)
        except Exception as e:
            raise DataExtractionError(f"OpenAI client initialization error: {str(e)}")

    @observe(name="openai_chat_completion", as_type="generation")
    def _chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Execute OpenAI chat completion request.
        
        Sends messages to OpenAI API and returns the response content.
        Configured with zero temperature for consistent outputs.
        
        Args:
            messages: List of message dictionaries for the conversation
            **kwargs: Additional parameters for the API call
            
        Returns:
            Generated response content as string
            
        Raises:
            DataExtractionError: If API call fails or returns empty response
        """
        try:
            response = self.cli.chat.completions.create(
                model=Config.OPENAI_MODEL, 
                messages=messages, 
                temperature=0, 
                **kwargs
            )
            if not response.choices or not response.choices[0].message.content:
                raise DataExtractionError("OpenAI returned empty response")
            
            return response.choices[0].message.content.strip()
            
        except OpenAIError as e:
            raise DataExtractionError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise DataExtractionError(f"Unexpected error during AI call: {str(e)}")

    @observe(name="discover_labels")
    def discover_labels(self, text: str, max_labels: int = 15) -> List[str]:
        """Discover potential field labels in document text.
        
        Uses AI to identify form field names and labels that appear
        in the document content. Useful for dynamic field discovery
        in unknown document formats.
        
        Args:
            text: Document text content to analyze
            max_labels: Maximum number of labels to return
            
        Returns:
            List of discovered field label strings
            
        Raises:
            DataExtractionError: If text is empty or analysis fails
        """
        if not text or not text.strip():
            raise DataExtractionError("No text content to analyze")
        
        try:
            prompt: str = (
                "Return comma-separated labels (no values) that look like form-field names "
                f"in the document below (≤{max_labels}).\n\n{text[:3000]}"
            )
            raw_response: str = self._chat(
                [
                    {"role": "system", "content": "You are PDF-data assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
            )
            
            if not raw_response:
                return []
            
            labels: List[str] = [
                label.strip() for label in raw_response.split(",") 
                if 2 < len(label.strip()) < 40
            ]
            return labels[:max_labels]
            
        except Exception as e:
            if isinstance(e, DataExtractionError):
                raise
            raise DataExtractionError(f"Label discovery error: {str(e)}")

    @observe(name="ai_extract_data")
    def extract(self, text: str, fields: List[str]) -> Dict[str, str]:
        """Extract specified fields using AI analysis.
        
        Uses OpenAI to analyze text content and extract values for
        the specified fields. Returns structured data as JSON.
        
        Args:
            text: Input text content to extract data from
            fields: List of field names to extract
            
        Returns:
            Dictionary mapping field names to extracted values
            
        Raises:
            DataExtractionError: If text is empty, no fields specified, or extraction fails
        """
        if not text or not text.strip():
            raise DataExtractionError("No text content to process")
        
        if not fields:
            raise DataExtractionError("No fields specified for extraction")
        
        try:
            prompt: str = self._build_extraction_prompt(fields, text)
            raw_response: str = self._chat(
                [
                    {"role": "system", "content": "You are data-extraction engine."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
            )
            
            return self._parse_extraction_result(raw_response)
                
        except Exception as e:
            if isinstance(e, DataExtractionError):
                raise
            raise DataExtractionError(f"AI extraction error: {str(e)}")
    
    def _build_extraction_prompt(self, fields: List[str], text: str) -> str:
        """Build prompt for AI extraction request.
        
        Constructs a detailed prompt that instructs the AI model
        to extract specific fields and return results in JSON format.
        
        Args:
            fields: List of field names to extract
            text: Document text content (truncated to 20k characters)
            
        Returns:
            Formatted prompt string for AI model
        """
        return (
            f"Extract: {', '.join(fields)}\n\n"
            "Return ONLY compact JSON {\"Field\":\"Value\"}. "
            "If a field is missing, set null.\n\n"
            + text[:20_000]
        )
    
    def _parse_extraction_result(self, raw_response: str) -> Dict[str, str]:
        """Parse AI response to extract JSON data.
        
        Locates JSON content in the AI response and parses it
        into a dictionary structure for field-value mapping.
        
        Args:
            raw_response: Raw response text from AI model
            
        Returns:
            Parsed dictionary containing extracted field-value pairs
            
        Raises:
            DataExtractionError: If JSON parsing fails or format is invalid
        """
        match = re.search(r"\{.*\}", raw_response, re.S)
        if not match:
            raise DataExtractionError("AI did not return valid JSON")
        
        try:
            result: Any = json.loads(match.group(0))
            if not isinstance(result, dict):
                raise DataExtractionError("AI returned invalid data format")
            
            return result
            
        except json.JSONDecodeError as e:
            raise DataExtractionError(f"JSON parsing error from AI response: {str(e)}")

# ──────────────────────── 6. PDF Processing Service ────────────────────────

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

# ──────────────────────── 7. UI Components ────────────────────────

class FieldSelector:
    """UI component for field selection interfaces.
    
    This class provides reusable methods for rendering field selection
    checkboxes in the Streamlit interface. Supports both predefined
    regex fields and dynamically discovered AI fields.
    """
    
    @staticmethod
    def render_field_checkboxes(fields_config: Dict[str, Dict[str, Union[str, List[str]]]], 
                               key_prefix: str) -> List[str]:
        """Render checkboxes for predefined fields and return selected ones.
        
        Creates a grid layout of checkboxes for field selection.
        All fields are selected by default for user convenience.
        
        Args:
            fields_config: Dictionary containing field definitions with display names
            key_prefix: Unique prefix for checkbox keys to avoid conflicts
            
        Returns:
            List of selected field names
        """
        cols = st.columns(3)
        selected_fields: List[str] = []
        field_keys: List[str] = list(fields_config.keys())
        
        for i, field_key in enumerate(field_keys):
            display_name: str = fields_config[field_key]['display']
            if cols[i % 3].checkbox(display_name, True, key=f"{key_prefix}_{field_key}"):
                selected_fields.append(field_key)
        
        return selected_fields
    
    @staticmethod
    def render_ai_field_checkboxes(labels: List[str], key_prefix: str) -> List[str]:
        """Render checkboxes for AI-discovered fields and return selected ones.
        
        Creates a grid layout of checkboxes for dynamically discovered fields.
        All fields are selected by default for user convenience.
        
        Args:
            labels: List of field labels discovered by AI
            key_prefix: Unique prefix for checkbox keys to avoid conflicts
            
        Returns:
            List of selected field labels
        """
        cols = st.columns(3)
        selected_fields: List[str] = []
        
        for i, label in enumerate(labels):
            if cols[i % 3].checkbox(label, True, key=f"{key_prefix}_{label}"):
                selected_fields.append(label)
        
        return selected_fields

class UIRenderer:
    """Renders the main Streamlit user interface.
    
    This class handles all UI rendering operations including headers,
    mode selection, file upload interfaces, and result display.
    It coordinates the overall user experience flow.
    
    Attributes:
        field_selector: FieldSelector instance for field selection UI
    """
    
    def __init__(self) -> None:
        """Initialize UI renderer with field selector component."""
        self.field_selector: FieldSelector = FieldSelector()
    
    def render_header(self) -> None:
        """Render application header and configuration status.
        
        Sets up the page configuration, displays the main title,
        and shows Langfuse monitoring status information.
        """
        st.set_page_config(page_title="PDF Extractor", layout="wide")
        st.title("📄 PDF Extractor — Classic vs AI")
        
        if self._check_langfuse_config():
            st.info("🔍 Langfuse monitoring active")
        else:
            st.info("📊 Monitoring disabled - add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    
    def render_mode_selector(self) -> str:
        """Render processing mode selection radio buttons.
        
        Provides user interface for selecting between Classic (regex)
        and AI (GPT) extraction modes.
        
        Returns:
            Selected mode string
        """
        return st.radio("Parser mode:", ["Classic (Regex)", "AI (GPT-3.5)"], horizontal=True)
    
    def render_batch_section(self) -> Tuple[List[Any], List[str]]:
        """Render batch processing section interface.
        
        Displays file uploader for multiple files and field selection
        interface for batch processing operations.
        
        Returns:
            Tuple containing uploaded files list and selected fields list
        """
        st.header("📦 Batch PDF Extraction (async database saving)")
        
        uploaded_files: List[Any] = st.file_uploader(
            "Upload multiple PDF files",
            type="pdf",
            accept_multiple_files=True,
            key="batch_uploader"
        )
        
        selected_fields: List[str] = []
        if uploaded_files:
            st.subheader("🔤 Available fields for batch extraction")
            selected_fields = self.field_selector.render_field_checkboxes(
                Config.REGEX_FIELDS, "batch"
            )
        
        return uploaded_files, selected_fields
    
    def render_single_file_uploader(self) -> Any:
        """Render single file upload interface.
        
        Provides file uploader widget for single PDF file processing.
        
        Returns:
            Uploaded file object or None if no file uploaded
        """
        return st.file_uploader("Upload PDF", type="pdf", key="single_uploader")
    
    def render_extraction_results(self, data: Dict[str, str], filename: str, 
                                file_hash: str, db_id: int) -> None:
        """Render extraction results display interface.
        
        Shows extracted field-value pairs, database save confirmation,
        and provides download button for JSON results.
        
        Args:
            data: Extracted field-value pairs
            filename: Original PDF filename
            file_hash: SHA256 hash of file content
            db_id: Database ID of saved extraction record
        """
        st.success("Extraction completed")
        
        for key, value in data.items():
            display_name: str = Config.REGEX_FIELDS.get(key, {}).get('display', key)
            st.write(f"**{display_name}**: {value}")
        
        st.success(f"✅ Data saved to database (ID: {db_id})")
        
        st.download_button(
            "💾 Download JSON",
            json.dumps(data, ensure_ascii=False, indent=2),
            file_name=f"{Path(filename).stem}_{file_hash}.json",
            mime="application/json",
        )
    
    def _check_langfuse_config(self) -> bool:
        """Check if Langfuse monitoring is properly configured.
        
        Verifies that required environment variables for Langfuse
        monitoring are present and non-empty.
        
        Returns:
            True if Langfuse is properly configured, False otherwise
        """
        required_keys: List[str] = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
        missing_keys: List[str] = [key for key in required_keys if not os.getenv(key)]
        return len(missing_keys) == 0

# ──────────────────────── 8. Main Application ────────────────────────

class PDFExtractorApp:
    """Main Streamlit application class.
    
    This class orchestrates the entire application flow, coordinates
    all components, and handles the main user interaction logic.
    It serves as the application entry point and controller.
    
    Attributes:
        db_manager: DatabaseManager for database operations
        repository: ExtractionRepository for data persistence
        processor: PDFProcessor for file processing operations
        batch_processor: BatchProcessor for batch operations
        ui: UIRenderer for user interface operations
    """
    
    def __init__(self) -> None:
        """Initialize application with all required components.
        
        Sets up dependency injection for all major components
        and establishes the application architecture.
        """
        self.db_manager: DatabaseManager = DatabaseManager()
        self.repository: ExtractionRepository = ExtractionRepository(self.db_manager)
        self.processor: PDFProcessor = PDFProcessor(self.repository)
        self.batch_processor: BatchProcessor = BatchProcessor(self.processor)
        self.ui: UIRenderer = UIRenderer()
    
    def run(self) -> None:
        """Run the main application flow.
        
        Orchestrates the complete user interface flow including
        header rendering, mode selection, and processing workflows.
        Handles early exit conditions for missing configuration.
        """
        self.ui.render_header()
        
        mode: str = self.ui.render_mode_selector()
        
        if mode.startswith("AI") and not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ Set OPENAI_API_KEY environment variable or add it to .secrets.toml")
            return
        
        # Batch processing section
        self._handle_batch_processing(mode)
        
        # Single file processing section
        self._handle_single_file_processing(mode)
    
    def _handle_batch_processing(self, mode: str) -> None:
        """Handle batch processing workflow.
        
        Manages the complete batch processing user interface
        and workflow including file upload, field selection,
        and batch execution.
        
        Args:
            mode: Selected processing mode string
        """
        uploaded_files, selected_fields = self.ui.render_batch_section()
        
        if uploaded_files and selected_fields:
            if st.button("Extract All (async batch & save to DB)"):
                try:
                    extractor: DataExtractor = self._create_extractor(mode)
                    method: str = "classic" if mode.startswith("Classic") else "ai"
                    
                    batch_results: List[Dict[str, Any]] = self.batch_processor.process_batch(
                        uploaded_files, extractor, selected_fields, method
                    )
                    
                    self._display_batch_results(batch_results)
                    
                except Exception as e:
                    st.error(f"❌ Batch processing error: {str(e)}")
    
    def _handle_single_file_processing(self, mode: str) -> None:
        """Handle single file processing workflow.
        
        Manages the complete single file processing user interface
        and workflow, delegating to mode-specific handlers.
        
        Args:
            mode: Selected processing mode string
        """
        uploaded_file: Any = self.ui.render_single_file_uploader()
        if not uploaded_file:
            return
        
        try:
            if mode.startswith("Classic"):
                self._handle_classic_mode(uploaded_file)
            else:
                self._handle_ai_mode(uploaded_file)
                
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
    
    def _handle_classic_mode(self, uploaded_file: Any) -> None:
        """Handle classic regex extraction mode workflow.
        
        Manages the user interface and processing flow for
        regex-based extraction including field selection.
        
        Args:
            uploaded_file: Uploaded file object from Streamlit
        """
        st.header("🔤 Available fields (regex)")
        
        selected_fields: List[str] = self.ui.field_selector.render_field_checkboxes(
            Config.REGEX_FIELDS, "classic"
        )
        
        if st.button("Extract selected fields", key="single_extract_button"):
            self._process_single_file(uploaded_file, "Classic", selected_fields)
    
    def _handle_ai_mode(self, uploaded_file: Any) -> None:
        """Handle AI extraction mode workflow.
        
        Manages the user interface and processing flow for
        AI-based extraction including dynamic field discovery.
        
        Args:
            uploaded_file: Uploaded file object from Streamlit
        """
        pdf_bytes: bytes = uploaded_file.read()
        PDFValidator.validate_pdf_file(pdf_bytes, uploaded_file.name)
        text: str = TextExtractor.extract_text(pdf_bytes)
        
        extractor: AIExtractor = AIExtractor(os.getenv("OPENAI_API_KEY"))
        st.header("🤖 AI Quick-scan")
        
        with st.spinner("Scanning document..."):
            labels: List[str] = extractor.discover_labels(text)
        
        if not labels:
            st.warning("Model did not identify any labels.")
            return
        
        selected_fields: List[str] = self.ui.field_selector.render_ai_field_checkboxes(
            labels, "ai_field"
        )
        
        if st.button("Extract selected fields", key="single_ai_extract_button"):
            try:
                file_hash: str = sha256(pdf_bytes).hexdigest()[:6]
                data: Dict[str, str] = extractor.extract(text, selected_fields)
                
                if data:
                    db_id: int = self.processor.save_extraction_result(
                        uploaded_file.name, file_hash, "ai", data
                    )
                    self.ui.render_extraction_results(
                        data, uploaded_file.name, file_hash, db_id
                    )
                else:
                    st.warning("Failed to extract any data.")
                    
            except Exception as e:
                st.error(f"❌ AI extraction error: {str(e)}")
    
    def _process_single_file(self, uploaded_file: Any, mode: str, selected_fields: List[str]) -> None:
        """Process a single uploaded file with specified parameters.
        
        Executes the complete processing workflow for a single file
        including validation, extraction, and result display.
        
        Args:
            uploaded_file: Uploaded file object from Streamlit
            mode: Processing mode string
            selected_fields: List of fields to extract
        """
        if not selected_fields:
            st.warning("Select at least one field for extraction.")
            return
        
        try:
            with st.spinner("Extracting data..."):
                pdf_bytes: bytes = uploaded_file.read()
                extractor: DataExtractor = self._create_extractor(mode)
                
                data, file_hash = self.processor.process_file(
                    pdf_bytes, uploaded_file.name, extractor, selected_fields
                )
                
                if data:
                    method: str = "classic" if mode.startswith("Classic") else "ai"
                    db_id: int = self.processor.save_extraction_result(
                        uploaded_file.name, file_hash, method, data
                    )
                    self.ui.render_extraction_results(
                        data, uploaded_file.name, file_hash, db_id
                    )
                else:
                    st.warning("None of the selected fields were found.")
                    
        except Exception as e:
            st.error(f"❌ Extraction error: {str(e)}")
    
    def _create_extractor(self, mode: str) -> DataExtractor:
        """Factory method for creating appropriate extractor instances.
        
        Creates and returns the appropriate extractor implementation
        based on the selected processing mode.
        
        Args:
            mode: Processing mode string
            
        Returns:
            Appropriate DataExtractor implementation instance
        """
        if mode.startswith("Classic"):
            return ClassicExtractor()
        else:
            return AIExtractor(os.getenv("OPENAI_API_KEY"))
    
    def _display_batch_results(self, batch_results: List[Dict[str, Any]]) -> None:
        """Display batch processing results interface.
        
        Shows summary of batch processing results including
        successful extractions, errors, and download options.
        
        Args:
            batch_results: List of processing results for each file
        """
        st.success("✅ All files processed and saved!")
        st.subheader("Batch extraction results")
        
        for batch_result in batch_results:
            st.write(f"**File:** {batch_result['file']}")
            if "result" in batch_result:
                st.json(batch_result["result"])
                st.write(f"Saved to database (ID: {batch_result['db_id']})")
            else:
                st.error(f"Error: {batch_result['error']}")
        
        st.download_button(
            "💾 Download batch JSON",
            json.dumps(batch_results, ensure_ascii=False, indent=2),
            file_name="batch_results.json",
            mime="application/json",
        )

# ──────────────────────── 9. Entry Point ────────────────────────

if __name__ == "__main__":
    """Application entry point with error handling.
    
    Creates and runs the main application instance with
    top-level exception handling for critical errors.
    """
    try:
        app: PDFExtractorApp = PDFExtractorApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
