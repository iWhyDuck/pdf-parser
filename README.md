# PDF Parser Pro

A comprehensive PDF document analysis application with AI-powered field extraction, built with Streamlit and modern Python libraries.

## 🚀 Features

### Core Functionality
- **Multi-format PDF Processing**: Supports text extraction using PyPDF2, pdfplumber, and PyMuPDF with automatic fallback
- **Dual Processing Modes**: 
  - **Basic Mode**: Pattern-based field detection (no API key required)
  - **AI Mode**: Advanced AI-powered extraction using OpenAI GPT-4 and Vision APIs
- **OCR Support**: Automatic OCR fallback for image-based PDFs using OpenAI Vision API
- **Batch Processing**: Handle multiple PDF files simultaneously
- **Field Detection**: Automatic detection of extractable fields in documents
- **Database Persistence**: SQLite database for storing processing history and results

### Supported Field Types
- Email addresses
- Phone numbers
- Dates (multiple formats)
- Monetary amounts
- Postal codes (Polish and US formats)
- Tax numbers (NIP, PESEL, REGON)
- Invoice/order numbers
- Custom labeled fields (key: value pairs)

### Export Options
- JSON format download
- CSV format download
- Batch export for multiple documents
- Historical data retrieval

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) OpenAI API key for AI mode features

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd pdf-parser
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run main.py
   ```

5. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`

## 🔧 Configuration

### API Key Setup (Optional - for AI Mode)
1. Get your OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Enter the API key in the sidebar when using AI mode
3. The key is used for:
   - Advanced field detection
   - OCR processing of image-based PDFs
   - Context-aware data extraction

### Database Configuration
The application uses SQLite by default. The database file (`pdf_parser.db`) will be created automatically in the project directory.

## 📖 Usage Guide

### Basic Workflow

1. **Choose Processing Mode:**
   - **Basic Mode**: Fast, pattern-based extraction (recommended for structured documents)
   - **AI Mode**: Advanced AI extraction (requires OpenAI API key)

2. **Upload PDF Files:**
   - Support for multiple files (up to 5 files, 10MB each)
   - Automatic validation and processing

3. **Field Discovery:**
   - Application automatically detects extractable fields
   - Review and select fields of interest

4. **Extract Data:**
   - Process selected fields
   - View results in structured format

5. **Export Results:**
   - Download data as JSON or CSV
   - Access historical extractions

### Processing Modes Comparison

| Feature | Basic Mode | AI Mode |
|---------|------------|---------|
| API Key Required | ❌ No | ✅ Yes |
| Processing Speed | ⚡ Fast | 🐌 Slower |
| OCR Support | ❌ No | ✅ Yes |
| Complex Documents | 🟡 Limited | ✅ Excellent |
| Cost | 🆓 Free | 💰 Usage-based |
| Field Detection | Pattern-based | Context-aware |

### Best Practices

1. **File Preparation:**
   - Use high-quality PDF files for better results
   - Ensure text is searchable (not just images)
   - Consider file size limits (10MB max)

2. **Field Selection:**
   - Review detected fields before processing
   - Select only relevant fields to improve performance
   - Use descriptive field names for better AI understanding

3. **Performance Optimization:**
   - Use Basic mode for simple, structured documents
   - Reserve AI mode for complex or image-based PDFs
   - Process files in batches for efficiency

## 🏗️ Project Structure

```
pdf-parser/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── document.py          # Data models (Document, ExtractionJob, ExtractedField)
│   ├── database/
│   │   ├── __init__.py
│   │   └── manager.py           # Database operations and management
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py     # PDF text extraction with multiple libraries
│   │   ├── text_analyzer.py     # Pattern-based field detection
│   │   └── ai_extractor.py      # OpenAI API integration
│   ├── ui/
│   │   ├── __init__.py
│   │   └── components.py        # Streamlit UI components
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Utility functions and helpers
├── tests/
│   ├── __init__.py
│   ├── test_models.py           # Tests for data models
│   ├── test_database.py         # Tests for database operations
│   ├── test_processors.py       # Tests for PDF processing
│   └── test_utils.py            # Tests for utility functions
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🧪 Testing

The project includes comprehensive unit tests for all modules:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test module
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

### Test Coverage
- **Models**: Data class validation and integrity
- **Database**: CRUD operations, migrations, and data persistence
- **Processors**: PDF extraction, text analysis, and AI integration
- **Utils**: Helper functions and data conversion
- **Integration**: End-to-end workflow testing

## 🔍 API Reference

### Core Classes

#### `Document`
Represents a PDF document in the system.
```python
@dataclass
class Document:
    filename: str
    file_size: int
    file_hash: str
    processing_method: str
    processing_mode: str = "basic"
    id: Optional[int] = None
    upload_time: Optional[datetime] = None
```

#### `DatabaseManager`
Manages all database operations.
```python
class DatabaseManager:
    def save_document(self, document: Document) -> int
    def get_extraction_history(self, limit: int = 50) -> List[Dict]
    def save_extracted_fields(self, job_id: int, fields: Dict[str, Any]) -> None
```

#### `PDFProcessor`
Handles PDF text extraction with multiple fallback methods.
```python
class PDFProcessor:
    def extract_text_native(self, pdf_file) -> Tuple[Optional[str], str]
    def convert_pdf_to_images(self, pdf_file) -> List[Image.Image]
    def validate_pdf(self, pdf_file) -> Tuple[bool, str]
```

#### `BasicTextAnalyzer`
Pattern-based field detection and extraction.
```python
class BasicTextAnalyzer:
    def detect_basic_fields(self, text: str) -> List[str]
    def extract_basic_fields(self, text: str, selected_fields: List[str]) -> Dict[str, Any]
```

#### `AIExtractor`
AI-powered field detection and extraction using OpenAI APIs.
```python
class AIExtractor:
    def quick_scan_fields(self, text: str) -> List[str]
    def detailed_extraction(self, text: str, selected_fields: List[str]) -> Optional[Dict[str, Any]]
    def extract_text_from_images(self, images: List[Image.Image]) -> Optional[str]
```

## 🔧 Advanced Configuration

### Custom Field Patterns
You can extend the basic text analyzer with custom field patterns:

```python
# In src/processors/text_analyzer.py
self.field_patterns = {
    "Custom Field": r'your-regex-pattern-here',
    # Add more patterns...
}
```

### Database Customization
To use a different database location:

```python
from src.database import DatabaseManager

# Custom database path
db_manager = DatabaseManager("path/to/custom/database.db")
```

### API Configuration
Adjust AI model settings in `src/processors/ai_extractor.py`:

```python
TEXT_MODEL = "gpt-4-turbo"  # Change model as needed
VISION_MODEL = "gpt-4-vision-preview"
MAX_TOKENS_EXTRACTION = 1500  # Adjust token limits
```

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'src'" Error**
   - Ensure you're running from the project root directory
   - Check that `__init__.py` files exist in all module directories

2. **PDF Processing Fails**
   - Verify PDF file is not corrupted
   - Check file size limits (10MB max)
   - Ensure PDF is not password-protected

3. **AI Mode Not Working**
   - Verify OpenAI API key is valid and has sufficient credits
   - Check internet connectivity
   - Ensure API key has access to required models

4. **Database Errors**
   - Check file permissions in the project directory
   - Ensure SQLite is properly installed
   - Try deleting the database file to force recreation

5. **Streamlit Issues**
   - Update Streamlit to the latest version: `pip install -U streamlit`
   - Clear browser cache
   - Try running on a different port: `streamlit run main.py --server.port 8502`

### Performance Tips

1. **Memory Usage:**
   - Process large files individually rather than in batches
   - Close the application between heavy processing sessions

2. **Processing Speed:**
   - Use Basic mode for routine document processing
   - Reserve AI mode for complex or problematic documents

3. **API Costs:**
   - Monitor OpenAI usage in your dashboard
   - Use field selection to reduce API calls
   - Consider preprocessing documents to extract relevant sections

## 🤝 Contributing

### Development Setup

1. **Clone and setup development environment:**
   ```bash
   git clone <repository-url>
   cd pdf-parser
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Install development dependencies:**
   ```bash
   pip install pytest pytest-cov black flake8 mypy
   ```

3. **Run tests before making changes:**
   ```bash
   pytest --cov=src
   ```

4. **Code formatting:**
   ```bash
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

### Contribution Guidelines

1. **Code Style:**
   - Follow PEP 8 guidelines
   - Use type hints for all functions
   - Write comprehensive docstrings
   - Maintain test coverage above 80%

2. **Testing:**
   - Write tests for all new functionality
   - Update existing tests when modifying code
   - Ensure all tests pass before submitting

3. **Documentation:**
   - Update README for new features
   - Add docstrings to all public methods
   - Include usage examples for complex features

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙋 Support

### Getting Help

1. **Documentation**: Check this README and inline code documentation
2. **Issues**: Report bugs or request features via GitHub issues
3. **Testing**: Run the test suite to verify your installation

### Known Limitations

1. **File Size**: Maximum 10MB per PDF file
2. **Batch Processing**: Limited to 5 files per upload
3. **OCR**: Only available in AI mode with valid OpenAI API key
4. **Languages**: Optimized for English and Polish documents
5. **Complex Layouts**: Some complex PDF layouts may not extract perfectly

### Version History

- **v1.0.0**: Initial release with basic and AI processing modes
  - Multi-library PDF extraction
  - Pattern-based and AI field detection
  - SQLite database integration
  - Streamlit web interface
  - Comprehensive test suite

---

**PDF Parser Pro** - Streamlining document analysis with AI-powered extraction capabilities.