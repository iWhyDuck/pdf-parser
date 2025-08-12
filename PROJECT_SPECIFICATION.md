# PDF Parser - Project Specification

## Original Recruitment Task

> Build a simple application component / code snippet (please decide a good structure for your code) using Python and any open-source libraries, that extracts data elements (as 'key : value' pairs) from provided PDF forms and presents them to a user. The data elements should be configurable, but please choose a couple example elements to focus on. For example, given provided documents a data element might be 'customer name', 'branch name', 'claim type', or other. Create a small fit for purpose data model / tables to store relevant data required for the application to function & outputs. Create a simple UI (e.g. using Streamlit) to show the end-to-end process from document upload to extraction output being shown to the user. Consider extraction algorithm scalability, automated testing and error handling. We don't want perfection, but would like to see confidence, good practices and discussion on ways it could be improved. Please share the code with us only via a GitHub link.

## Core Project Requirements

### Functional Requirements
- **Output Format**: key:value pairs (all as text strings)
- **PDF Types**: Support all types (interactive forms, scanned documents, text-based PDFs)
- **User Interface**: Simple Streamlit-based web interface
- **Database**: SQLite with SQLAlchemy ORM for persistence
- **AI Integration**: OpenAI API for intelligent data extraction
- **Deployment**: Local development environment

### Key Architectural Decisions
- **AI Provider**: OpenAI API (GPT-4 for text, Vision API for OCR)
- **Processing Strategy**: Three-stage approach (native text extraction → OpenAI Vision OCR fallback → AI analysis)
- **Field Configuration**: AI suggests fields, user selects from list (no custom field creation)
- **Batch Processing**: Each PDF analyzed separately
- **API Strategy**: Variable calls (2-3 depending on PDF type)

## Application Workflow

### Main User Flow (Finalized)
1. **Upload**: User uploads one or multiple PDF files
2. **Text Processing**: System attempts native text extraction, falls back to OpenAI Vision API if needed
3. **Quick Scan**: AI performs rapid analysis to identify all possible extractable fields
4. **Field Selection**: System displays UNION of all fields from all PDFs as checkboxes (all checked by default)
5. **User Selection**: User unchecks unwanted fields
6. **Detailed Extraction**: AI performs thorough extraction of only selected fields for each PDF
7. **Results Display**: Display results in table/card format
8. **Export**: Option to export results to JSON/CSV

### Processing Strategy (Updated)

#### Three-Stage Processing Flow
```
Stage 1: Native Text Extraction
├── PyPDF2 → pdfplumber → pymupdf (fallback chain)
├── If ANY readable text found → proceed to Stage 3
└── If no text or insufficient text → proceed to Stage 2

Stage 2: OCR Fallback (OpenAI Vision API)
├── Convert PDF pages to images (pdf2image)
├── OpenAI Vision API: extract text from images
└── Proceed to Stage 3 with extracted text

Stage 3: AI Analysis Workflow (same for both paths)
├── Quick Scan: identify all possible fields (OpenAI text API)
├── User Field Selection via UI
└── Detailed Extraction: selected fields only (OpenAI text API)
```

#### API Call Patterns
- **Text-based PDFs**: 2 API calls (quick scan + detailed extraction)
- **Scanned/Image PDFs**: 3 API calls (Vision OCR + quick scan + detailed extraction)

#### Implementation Logic
```python
def process_pdf(pdf_file):
    # Stage 1: Try native text extraction
    text = extract_text_native(pdf_file)  # PyPDF2, pdfplumber, pymupdf
    
    # Stage 2: OCR fallback if needed
    if not text or len(text.strip()) < 10:  # Simple threshold
        images = convert_pdf_to_images(pdf_file)
        text = openai_vision_extract_text(images)
    
    # Stage 3: Standard AI workflow (consistent for both paths)
    return process_with_ai(text)
```

### Technical Workflow Details
- **Multiple PDFs**: Each PDF processed independently through full pipeline
- **Field Types**: All fields treated as text strings (simplification)
- **No Custom Fields**: Users can ONLY select from AI-suggested fields
- **Consistent UX**: Same field selection workflow regardless of PDF type
- **Error Handling**: Graceful degradation with clear error messages

## Technical Architecture

### Project Structure
```
pdf-parser/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # OpenAI API keys, database config
│   │   └── constants.py         # Prompts, error messages, thresholds
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py          # SQLAlchemy setup and models
│   │   └── schemas.py           # Pydantic validation models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py     # Native text extraction + image conversion
│   │   ├── ai_extractor.py      # OpenAI API interactions (text + vision)
│   │   └── data_service.py      # Business logic orchestration
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── extraction_repo.py   # Database operations
│   └── ui/
│       ├── __init__.py
│       ├── app.py              # Main Streamlit application
│       ├── components/          # Reusable UI components
│       │   ├── __init__.py
│       │   ├── upload.py       # File upload components
│       │   ├── results.py      # Results display components
│       │   └── progress.py     # Progress indicators
│       └── pages/              # Application pages
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### Technology Stack
- **Python 3.11+**
- **UI Framework**: Streamlit
- **PDF Processing**: PyPDF2, pdfplumber, pymupdf (multi-library fallback chain)
- **Image Processing**: pdf2image (PDF to images for Vision API)
- **OCR Processing**: OpenAI Vision API (images → text)
- **AI Analysis**: OpenAI GPT-4 API (text → structured extraction)
- **Database**: SQLite + SQLAlchemy ORM
- **Validation**: Pydantic models
- **Testing**: pytest framework
- **Environment**: python-dotenv for configuration

### Database Schema
1. **documents**
   - id (Primary Key, INTEGER)
   - filename (VARCHAR(255))
   - file_size (INTEGER)
   - upload_time (DATETIME)
   - file_hash (VARCHAR(64)) - SHA-256 for duplicate detection
   - processing_method (VARCHAR(20)) - 'native' or 'ocr'

2. **extraction_jobs**
   - id (Primary Key, INTEGER)
   - document_id (Foreign Key, INTEGER)
   - status (VARCHAR(20)) - 'pending'/'processing'/'completed'/'failed'
   - selected_fields (JSON) - fields chosen by user
   - total_fields_found (INTEGER) - number of fields discovered
   - created_at (DATETIME)
   - completed_at (DATETIME)
   - error_message (TEXT) - if status is 'failed'

3. **extracted_fields**
   - id (Primary Key, INTEGER)
   - job_id (Foreign Key, INTEGER)
   - field_name (VARCHAR(255))
   - field_value (TEXT)
   - confidence_score (FLOAT) - AI confidence level (0.0-1.0)

## AI Integration Strategy

### Prompt Templates

**Vision API OCR Prompt:**
```
Extract all text from this document image. 
Return the text exactly as it appears, preserving formatting and structure.
Focus on readable text and ignore decorative elements.
If the image contains forms, preserve the relationship between labels and values.
```

**Quick Scan Prompt:**
```
Analyze this document text and identify all possible data fields that could be extracted.
Return only field names as a simple comma-separated list.
Examples: Customer Name, Policy Number, Claim Date, Amount, Branch Office
Focus on structured data fields that appear to have specific values.
Do not include generic text, paragraphs, or headers.
Look for form fields, labeled data, and key-value pairs.
```

**Detailed Extraction Prompt:**
```
Extract the following specific fields from this document: {selected_fields}

Return as valid JSON with exact field_name: field_value pairs.
If a field is not found or unclear, return null for that field.
Extract values as plain text, preserve original formatting.
Be precise and only extract the actual value, not surrounding text or labels.

Expected format:
{
  "Customer Name": "John Smith",
  "Policy Number": "POL-123456",
  "Missing Field": null
}
```

### Error Handling Strategy
- **Native Text Extraction Failures**: Sequential fallback through PDF libraries
- **Vision API Failures**: Retry with exponential backoff, clear error messages
- **API Rate Limits**: Queue management with retry logic
- **No Fields Found**: Informative message with suggestions for user
- **Partial Extraction**: Display successful fields + list failed ones
- **Invalid PDF Files**: Clear validation errors with format requirements
- **Network Issues**: Offline mode indicators and retry mechanisms

## Services Architecture

### PDFProcessor Service
```python
class PDFProcessor:
    def extract_text_native(pdf_file) -> str | None
    def convert_to_images(pdf_file) -> List[Image]
    def detect_text_quality(text: str) -> bool
```

### AIExtractor Service
```python
class AIExtractor:
    def extract_text_from_images(images: List[Image]) -> str
    def quick_scan(text: str) -> List[str]
    def detailed_extraction(text: str, fields: List[str]) -> Dict[str, str]
```

### DataService (Orchestration)
```python
class DataService:
    def process_pdf(pdf_file) -> ExtractionResult
    def batch_process(pdf_files: List) -> List[ExtractionResult]
    def save_results(job_id: int, results: Dict) -> None
```

## User Interface Design

### Main Application Pages
1. **Home/Upload Page**
   - Drag & drop file upload (multiple files supported)
   - File validation and preview
   - Processing progress indicators

2. **Field Selection Page**
   - Checkbox interface for field selection
   - Preview of detected fields with sample values
   - Bulk select/deselect options

3. **Results Page**
   - Tabular view of extracted data
   - JSON view toggle for technical users
   - Export functionality (CSV, JSON, Excel)

4. **History Page**
   - Previous extraction jobs
   - Re-download results
   - Job status monitoring

### UI Components
- **Progress Indicators**: Real-time processing status with stages
- **Error Handling**: Toast notifications and detailed error panels
- **Loading States**: Spinners with descriptive status messages
- **Responsive Design**: Works on desktop and tablet devices
- **Accessibility**: Proper ARIA labels and keyboard navigation

## Implementation Plan

### Phase 1: Foundation Setup (Day 1)
1. Initialize project structure with all directories
2. Setup SQLAlchemy database models and migrations
3. Configuration management with environment variables
4. Basic requirements.txt and dependency management
5. Initialize git repository with proper .gitignore
6. **Documentation Setup**: Initialize README.md, setup documentation structure
7. **Testing Framework**: Setup pytest, testing directory structure, and CI configuration

### Phase 2: Core Services Development (Day 1-2)
1. PDF processor with multi-library text extraction
2. Image conversion utilities (pdf2image integration)
3. Mock AI extractor service (for testing without API costs)
4. Database repository pattern implementation
5. Data service orchestration layer
6. **Service Documentation**: Comprehensive docstrings for all service classes and methods
7. **Unit Tests**: Complete test coverage for PDFProcessor, AIExtractor, and DataService
8. **Integration Tests**: Database operations and service orchestration testing

### Phase 3: AI Integration (Day 2)
1. OpenAI API client setup and authentication
2. Vision API integration for OCR processing
3. Text API integration for field detection and extraction
4. Prompt engineering and response parsing
5. Error handling and retry mechanisms
6. **API Documentation**: Document all AI service interactions and prompt templates
7. **Mock Testing**: Comprehensive testing with mocked AI responses
8. **Error Scenario Tests**: API failures, rate limits, and malformed responses

### Phase 4: User Interface Development (Day 2-3)
1. Basic Streamlit application structure
2. File upload interface with drag & drop
3. Progress tracking and status display
4. Field selection interface with checkboxes
5. Results display with table and JSON views
6. **UI Documentation**: User interface components and interaction flow documentation
7. **Component Tests**: Individual UI component testing
8. **End-to-End Tests**: Complete user workflow testing

### Phase 5: Testing and Polish (Day 3-4)
1. **Comprehensive Testing Suite**: 
   - Unit tests achieving >85% coverage (>95% for core processing)
   - Integration tests for complete workflows
   - Performance tests for concurrent processing
   - Load testing with large files
2. **Error Scenario Testing**: Network failures, corrupted files, API limits
3. **Documentation Completion**:
   - Complete API documentation with OpenAPI/Swagger
   - Comprehensive README with setup and deployment instructions
   - Architecture diagrams and system design documentation
   - Configuration guide and troubleshooting section
4. **Code Quality Assurance**:
   - Code review and refactoring
   - Type hints validation
   - Consistent formatting and style
5. **UI/UX Polish**: Improvements and styling
6. **Export Functionality**: File downloads and format validation
7. **Performance Optimization**: Memory usage and processing speed
8. **Deployment Documentation**: Production setup and monitoring guidelines

## Testing Strategy

### Unit Testing
- **PDF Processing**: Test each library's text extraction
- **AI Response Parsing**: Mock API responses and JSON parsing
- **Database Operations**: CRUD operations and data integrity
- **Utility Functions**: File handling, validation, formatting
- **Service Classes**: Individual method testing with proper mocking
- **Data Models**: Validation logic and data transformation methods

### Integration Testing
- **End-to-End Workflow**: Complete PDF processing pipeline
- **Database Transactions**: Multi-table operations
- **API Error Scenarios**: Rate limits, network failures, invalid responses
- **File Upload Edge Cases**: Large files, corrupted PDFs, invalid formats
- **Service Orchestration**: Testing interaction between DataService, PDFProcessor, and AIExtractor
- **Configuration Management**: Environment variable handling and feature flags

### UI Testing
- **Component Behavior**: File uploads, field selection, results display
- **Error State Handling**: Display of error messages and recovery options
- **Export Functionality**: File downloads and format validation
- **Responsive Behavior**: Different screen sizes and user interactions

### Test Coverage Requirements
- **Minimum Coverage**: 85% overall code coverage
- **Critical Components**: 95% coverage for core processing logic
- **Test Documentation**: Each test case with clear description and purpose
- **Mock Strategy**: Comprehensive mocking for external dependencies (AI APIs, file system)
- **Automated Testing**: CI/CD integration with automated test execution
- **Performance Tests**: Load testing for concurrent file processing

## Success Criteria

### Core Functionality
- ✅ Successfully process text-based and scanned PDF documents
- ✅ AI-powered field detection with high accuracy (>85%)
- ✅ Intuitive field selection interface
- ✅ Reliable data storage and retrieval
- ✅ Multiple export formats (JSON, CSV)
- ✅ Comprehensive error handling with user guidance

### Code Quality Standards
- ✅ Clean architecture with clear separation of concerns
- ✅ Test coverage >85% for critical components (>95% for core processing)
- ✅ Proper error handling and logging throughout
- ✅ Type hints and input validation on all functions
- ✅ Comprehensive documentation and inline comments
- ✅ Consistent code style and formatting

### Documentation Requirements
- ✅ **Module Documentation**: Every module with clear purpose and usage examples
- ✅ **Class Documentation**: Detailed docstrings for all classes with attributes and methods
- ✅ **Function Documentation**: All functions with docstrings including parameters, return types, and examples
- ✅ **API Documentation**: Complete OpenAPI/Swagger documentation for all endpoints
- ✅ **README Files**: Comprehensive setup, usage, and deployment instructions
- ✅ **Code Comments**: Complex logic explained with inline comments
- ✅ **Architecture Documentation**: System design and component interaction diagrams
- ✅ **Configuration Guide**: Environment setup and configuration options explained
- ✅ **Error Handling Documentation**: Expected exceptions and recovery strategies
- ✅ **Testing Documentation**: How to run tests and interpret results

### User Experience Goals
- ✅ Intuitive workflow requiring minimal instructions
- ✅ Clear progress indicators and status feedback
- ✅ Helpful error messages with actionable guidance
- ✅ Response times <10 seconds for most operations
- ✅ Reliable processing of various PDF types and sizes

### Performance Targets
- ✅ Handle PDFs up to 10MB efficiently
- ✅ Process multiple files in reasonable time
- ✅ Minimal memory usage during processing
- ✅ Graceful handling of API rate limits

## Future Enhancement Opportunities

### Advanced AI Capabilities
- **Document Type Detection**: Automatic template selection for common forms
- **Confidence Scoring**: Visual indicators for extraction reliability
- **Learning from Corrections**: Improve accuracy based on user feedback
- **Multi-language Support**: Process documents in various languages

### Scalability Improvements
- **Async Processing**: Background job processing with Celery/Redis
- **Caching Layer**: Redis for frequently accessed data
- **Cloud Deployment**: Docker containers for AWS/GCP deployment
- **Horizontal Scaling**: Load balancing for multiple instances

### Enhanced User Experience
- **PDF Preview**: Visual document display with highlighted fields
- **Batch Operations**: Advanced batch processing dashboard
- **User Management**: Authentication and user-specific histories
- **Template System**: Save and reuse field configurations

### Integration Features
- **REST API**: Programmatic access for external systems
- **Webhooks**: Real-time notifications for processing completion
- **Cloud Storage**: Integration with S3, Google Drive, Dropbox
- **Enterprise Features**: SSO, audit trails, compliance reporting

## Risk Mitigation

### Technical Risks
- **OpenAI API Changes**: Abstract API interactions with adapter pattern
- **PDF Processing Failures**: Multiple fallback strategies with clear error reporting
- **Database Performance**: Proper indexing and query optimization
- **Memory Usage**: Stream processing for large files, proper garbage collection

### Business Risks
- **API Costs**: Usage monitoring, budget alerts, and cost optimization
- **Data Security**: Secure file handling, no persistent storage of sensitive data
- **User Adoption**: Intuitive design with comprehensive documentation
- **Maintenance**: Well-structured code with comprehensive test coverage

## Configuration Management

### Environment Variables
```
# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL_TEXT=gpt-4-turbo
OPENAI_MODEL_VISION=gpt-4-vision-preview

# Database Configuration
DATABASE_URL=sqlite:///pdf_parser.db

# Application Configuration
MAX_FILE_SIZE_MB=10
MAX_FILES_PER_UPLOAD=5
TEXT_EXTRACTION_MIN_LENGTH=10

# Development Settings
DEBUG=False
LOG_LEVEL=INFO
```

### Feature Flags
- **ENABLE_OCR_FALLBACK**: Toggle Vision API usage
- **ENABLE_BATCH_PROCESSING**: Multiple file uploads
- **ENABLE_EXPORT_FORMATS**: Different export options
- **ENABLE_PROCESSING_HISTORY**: Job history tracking

---

**Document Version**: 2.0  
**Last Updated**: January 2024  
**Status**: Ready for Implementation  
**Estimated Timeline**: 4 working days  
**Target Completion**: Full MVP with core functionality and testing