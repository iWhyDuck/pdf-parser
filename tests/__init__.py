"""Test package for PDF Parser application.

This package contains comprehensive unit tests for all components
of the PDF parser system including validators, extractors, database
operations, and processors.

Test Structure:
- conftest.py: Shared fixtures and test configuration
- test_validators.py: Tests for PDF validation functionality
- test_extractors.py: Tests for text and data extraction
- test_database.py: Tests for database operations
- test_processors.py: Tests for processing workflows

Usage:
    Run all tests: pytest
    Run specific module: pytest tests/test_extractors.py
    Run with coverage: pytest --cov=src/pdf_parser
"""

__version__ = "1.0.0"
