#!/usr/bin/env python3
"""Validation script to test the PDF parser setup and basic functionality.

This script performs basic checks to ensure the application is properly
configured and all components can be imported and initialized correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_imports():
    """Test that all modules can be imported correctly."""
    print("🔍 Testing module imports...")

    try:
        from pdf_parser import (
            Config, PDFProcessingError, DataExtractionError, DatabaseError,
            ValidationError, Base, Extraction, DatabaseManager,
            ExtractionRepository, PDFValidator, TextExtractor,
            ClassicExtractor, AIExtractor, PDFProcessor, BatchProcessor,
            FieldSelector, UIRenderer
        )
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def check_configuration():
    """Test configuration loading."""
    print("\n🔧 Testing configuration...")

    try:
        from pdf_parser import Config

        assert hasattr(Config, 'OPENAI_MODEL')
        assert hasattr(Config, 'MAX_FILE_SIZE')
        assert hasattr(Config, 'MIN_FILE_SIZE')
        assert hasattr(Config, 'DATABASE_URL')
        assert hasattr(Config, 'REGEX_FIELDS')
        assert len(Config.REGEX_FIELDS) > 0

        print(f"✅ Configuration loaded (OpenAI Model: {Config.OPENAI_MODEL})")
        print(f"   Max file size: {Config.MAX_FILE_SIZE // (1024*1024)}MB")
        print(f"   Available fields: {list(Config.REGEX_FIELDS.keys())}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_database():
    """Test database connectivity and model creation."""
    print("\n🗄️ Testing database connectivity...")

    try:
        from pdf_parser import DatabaseManager, ExtractionRepository

        # Use in-memory SQLite for testing
        db_manager = DatabaseManager("sqlite:///:memory:")

        # Test engine creation
        engine = db_manager.engine
        print("✅ Database engine created successfully")

        # Test session creation
        session = db_manager.create_session()
        session.close()
        print("✅ Database session created successfully")

        # Test repository
        repo = ExtractionRepository(db_manager)
        test_data = {"test_field": "test_value"}
        db_id = repo.save_extraction("test.pdf", "hash123", "test", test_data)
        print(f"✅ Repository save test successful (ID: {db_id})")

        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def check_extractors():
    """Test extractor initialization."""
    print("\n🔎 Testing extractors...")

    try:
        from pdf_parser import ClassicExtractor, TextExtractor

        # Test classic extractor
        classic = ClassicExtractor()
        print("✅ Classic extractor initialized")

        # Test text extractor
        text_extractor = TextExtractor()
        print("✅ Text extractor initialized")

        # Test AI extractor (if API key available)
        if os.getenv("OPENAI_API_KEY"):
            from pdf_parser import AIExtractor
            ai = AIExtractor(os.getenv("OPENAI_API_KEY"))
            print("✅ AI extractor initialized with API key")
        else:
            print("⚠️ No OpenAI API key found - AI extractor not tested")

        return True
    except Exception as e:
        print(f"❌ Extractor error: {e}")
        return False

def check_validators():
    """Test PDF validation functionality."""
    print("\n🛡️ Testing validators...")

    try:
        from pdf_parser import PDFValidator

        # Test with valid PDF bytes
        valid_pdf = b"%PDF-1.4" + b"x" * 200  # Minimal valid PDF
        PDFValidator.validate_pdf_file(valid_pdf, "test.pdf")
        print("✅ PDF validation successful")

        return True
    except Exception as e:
        print(f"❌ Validator error: {e}")
        return False

def check_processors():
    """Test processor initialization."""
    print("\n⚙️ Testing processors...")

    try:
        from pdf_parser import PDFProcessor, BatchProcessor, DatabaseManager, ExtractionRepository

        db_manager = DatabaseManager("sqlite:///:memory:")
        repo = ExtractionRepository(db_manager)

        # Test PDF processor
        pdf_processor = PDFProcessor(repo)
        print("✅ PDF processor initialized")

        # Test batch processor
        batch_processor = BatchProcessor(pdf_processor)
        print("✅ Batch processor initialized")

        return True
    except Exception as e:
        print(f"❌ Processor error: {e}")
        return False

def check_environment():
    """Check environment variables and dependencies."""
    print("\n🌍 Checking environment...")

    # Check Python version
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} - Need Python 3.8+")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Check key dependencies
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not installed")
        return False

    try:
        import pdfplumber
        print(f"✅ pdfplumber available")
    except ImportError:
        print("❌ pdfplumber not installed")
        return False

    try:
        import sqlalchemy
        print(f"✅ SQLAlchemy {sqlalchemy.__version__}")
    except ImportError:
        print("❌ SQLAlchemy not installed")
        return False

    # Check optional dependencies
    try:
        import openai
        print(f"✅ OpenAI {openai.__version__}")
    except ImportError:
        print("⚠️ OpenAI library not installed - AI features unavailable")

    # Check environment variables
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY configured")
    else:
        print("⚠️ OPENAI_API_KEY not set - AI features will not work")

    return True

def main():
    """Run all validation checks."""
    print("🚀 PDF Parser Setup Validation")
    print("=" * 50)

    checks = [
        check_environment,
        check_imports,
        check_configuration,
        check_database,
        check_extractors,
        check_validators,
        check_processors
    ]

    passed = 0
    total = len(checks)

    for check in checks:
        try:
            if check():
                passed += 1
        except Exception as e:
            print(f"❌ Unexpected error in {check.__name__}: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Validation Results: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All checks passed! Your setup is ready.")
        print("\nTo start the application, run:")
        print("    streamlit run src/app.py")
        return 0
    else:
        print("⚠️ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
