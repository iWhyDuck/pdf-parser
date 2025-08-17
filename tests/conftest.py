"""Pytest configuration and fixtures for the PDF parser test suite.

This module provides shared fixtures and test configuration for all test modules.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.pdf_parser import (
    Config,
    DatabaseManager,
    ExtractionRepository,
    PDFValidator,
    ClassicExtractor,
    TextExtractor,
    Base
)


@pytest.fixture(scope="session")
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    yield f"sqlite:///{db_path}"

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def test_db_manager(temp_db_path: str) -> DatabaseManager:
    """Create a DatabaseManager instance with a temporary database."""
    return DatabaseManager(database_url=temp_db_path)


@pytest.fixture
def test_repository(test_db_manager: DatabaseManager) -> ExtractionRepository:
    """Create an ExtractionRepository with a test database."""
    return ExtractionRepository(test_db_manager)


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Create sample PDF bytes for testing."""
    # Minimal valid PDF content
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Resources <<
/Font <<
/F1 4 0 R
>>
>>
/Contents 5 0 R
>>
endobj

4 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj

5 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Customer Name: John Doe) Tj
ET
endstream
endobj

xref
0 6
0000000000 65535 f
0000000010 00000 n
0000000079 00000 n
0000000136 00000 n
0000000297 00000 n
0000000377 00000 n
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
472
%%EOF"""
    return pdf_content


@pytest.fixture
def sample_text_content() -> str:
    """Sample text content extracted from PDF."""
    return """
Customer Name: John Smith
Policy Number: POL-123456
Claim Amount: $1,500.00
Date: 2024-01-01
"""


@pytest.fixture
def invalid_pdf_bytes() -> bytes:
    """Invalid PDF bytes for testing validation."""
    return b"This is not a PDF file" + b"x" * Config.MIN_FILE_SIZE


@pytest.fixture
def too_large_pdf_bytes() -> bytes:
    """PDF bytes that exceed size limit."""
    return b"%PDF-1.4" + b"x" * (Config.MAX_FILE_SIZE + 1)


@pytest.fixture
def too_small_pdf_bytes() -> bytes:
    """PDF bytes that are below minimum size."""
    return b"x" * (Config.MIN_FILE_SIZE - 1)


@pytest.fixture
def classic_extractor() -> ClassicExtractor:
    """Create a ClassicExtractor instance for testing."""
    return ClassicExtractor()


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing AI extractor."""
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = '{"customer_name": "John Doe", "policy_number": "POL-123"}'
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_extraction_data() -> Dict[str, Any]:
    """Sample extraction data for testing."""
    return {
        "customer_name": "John Smith",
        "policy_number": "POL-123456",
        "claim_amount": "$1,500.00"
    }


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration with sample regex fields."""
    return {
        "test_field": {
            "display": "Test Field",
            "patterns": [r"Test Field[:\s]*([^\n\r]+)"]
        },
        "number_field": {
            "display": "Number Field",
            "patterns": [r"Number[:\s]*(\d+)"]
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Store original values
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    original_langfuse_public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    original_langfuse_secret = os.environ.get("LANGFUSE_SECRET_KEY")

    # Set test values
    os.environ["OPENAI_API_KEY"] = "test-key-123"
    os.environ["LANGFUSE_PUBLIC_KEY"] = "test-public"
    os.environ["LANGFUSE_SECRET_KEY"] = "test-secret"

    yield

    # Restore original values
    if original_openai_key is not None:
        os.environ["OPENAI_API_KEY"] = original_openai_key
    elif "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    if original_langfuse_public is not None:
        os.environ["LANGFUSE_PUBLIC_KEY"] = original_langfuse_public
    elif "LANGFUSE_PUBLIC_KEY" in os.environ:
        del os.environ["LANGFUSE_PUBLIC_KEY"]

    if original_langfuse_secret is not None:
        os.environ["LANGFUSE_SECRET_KEY"] = original_langfuse_secret
    elif "LANGFUSE_SECRET_KEY" in os.environ:
        del os.environ["LANGFUSE_SECRET_KEY"]
