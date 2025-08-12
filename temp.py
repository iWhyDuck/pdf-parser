"""
PDF Parser - Complete Single File Implementation
Full-featured PDF document analysis with AI extraction, database persistence, and batch processing
"""

import streamlit as st
import PyPDF2
import pdfplumber
import pymupdf  # fitz
import openai
import json
import io
import base64
import hashlib
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import traceback
from dataclasses import dataclass
from pathlib import Path
import tempfile
from PIL import Image
import os
from pdf2image import convert_from_bytes
import time

# Page Configuration
st.set_page_config(
    page_title="PDF Parser Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
OPENAI_MODEL_TEXT = "gpt-4-turbo"
OPENAI_MODEL_VISION = "gpt-4-vision-preview"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_FILES_PER_UPLOAD = 5
TEXT_EXTRACTION_MIN_LENGTH = 10
DATABASE_PATH = "pdf_parser.db"

# CSS Styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: #1f1f1f;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .results-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #fee;
        border: 1px solid #fcc;
        color: #c00;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #efe;
        border: 1px solid #cfc;
        color: #060;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .processing-stage {
        background-color: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 3px;
    }
    .field-selection {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Database Schema and Models
@dataclass
class Document:
    id: Optional[int] = None
    filename: str = ""
    file_size: int = 0
    upload_time: Optional[datetime] = None
    file_hash: str = ""
    processing_method: str = ""

@dataclass
class ExtractionJob:
    id: Optional[int] = None
    document_id: int = 0
    status: str = "pending"
    selected_fields: List[str] = None
    total_fields_found: int = 0
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: str = ""

@dataclass
class ExtractedField:
    id: Optional[int] = None
    job_id: int = 0
    field_name: str = ""
    field_value: str = ""
    confidence_score: float = 0.0

class DatabaseManager:
    """Manages SQLite database operations"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_hash TEXT UNIQUE NOT NULL,
                processing_method TEXT NOT NULL
            )
        """)
        
        # Extraction jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extraction_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                selected_fields TEXT,  -- JSON string
                total_fields_found INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """)
        
        # Extracted fields table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                field_name TEXT NOT NULL,
                field_value TEXT,
                confidence_score REAL DEFAULT 0.0,
                FOREIGN KEY (job_id) REFERENCES extraction_jobs (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_document(self, doc: Document) -> int:
        """Save document and return ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO documents 
            (filename, file_size, file_hash, processing_method)
            VALUES (?, ?, ?, ?)
        """, (doc.filename, doc.file_size, doc.file_hash, doc.processing_method))
        
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return doc_id
    
    def save_extraction_job(self, job: ExtractionJob) -> int:
        """Save extraction job and return ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        selected_fields_json = json.dumps(job.selected_fields) if job.selected_fields else None
        
        cursor.execute("""
            INSERT INTO extraction_jobs 
            (document_id, status, selected_fields, total_fields_found, error_message)
            VALUES (?, ?, ?, ?, ?)
        """, (job.document_id, job.status, selected_fields_json, 
              job.total_fields_found, job.error_message))
        
        job_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return job_id
    
    def update_extraction_job(self, job_id: int, status: str, error_message: str = ""):
        """Update extraction job status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        completed_at = datetime.now().isoformat() if status == 'completed' else None
        
        cursor.execute("""
            UPDATE extraction_jobs 
            SET status = ?, completed_at = ?, error_message = ?
            WHERE id = ?
        """, (status, completed_at, error_message, job_id))
        
        conn.commit()
        conn.close()
    
    def save_extracted_fields(self, job_id: int, fields: Dict[str, Any]):
        """Save extracted fields for a job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for field_name, field_value in fields.items():
            cursor.execute("""
                INSERT INTO extracted_fields 
                (job_id, field_name, field_value, confidence_score)
                VALUES (?, ?, ?, ?)
            """, (job_id, field_name, str(field_value), 1.0))  # Default confidence
        
        conn.commit()
        conn.close()
    
    def get_extraction_history(self, limit: int = 50) -> List[Dict]:
        """Get extraction job history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                ej.id, d.filename, ej.status, ej.total_fields_found,
                ej.created_at, ej.completed_at, d.processing_method
            FROM extraction_jobs ej
            JOIN documents d ON ej.document_id = d.id
            ORDER BY ej.created_at DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'job_id': row[0],
                'filename': row[1],
                'status': row[2],
                'fields_found': row[3],
                'created_at': row[4],
                'completed_at': row[5],
                'method': row[6]
            })
        
        conn.close()
        return results
    
    def get_extraction_results(self, job_id: int) -> Dict[str, str]:
        """Get extracted fields for a job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT field_name, field_value
            FROM extracted_fields
            WHERE job_id = ?
        """, (job_id,))
        
        results = {}
        for row in cursor.fetchall():
            results[row[0]] = row[1]
        
        conn.close()
        return results

class PDFProcessor:
    """Multi-stage PDF text extraction with fallbacks"""
    
    @staticmethod
    def get_file_hash(file_content: bytes) -> str:
        """Calculate SHA-256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()
    
    @staticmethod
    def extract_text_pypdf2(pdf_file) -> Optional[str]:
        """Extract text using PyPDF2"""
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            if pdf_reader.is_encrypted:
                return None
                
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip() if len(text.strip()) >= TEXT_EXTRACTION_MIN_LENGTH else None
            
        except Exception:
            return None
    
    @staticmethod
    def extract_text_pdfplumber(pdf_file) -> Optional[str]:
        """Extract text using pdfplumber"""
        try:
            pdf_file.seek(0)
            
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            return text.strip() if len(text.strip()) >= TEXT_EXTRACTION_MIN_LENGTH else None
            
        except Exception:
            return None
    
    @staticmethod
    def extract_text_pymupdf(pdf_file) -> Optional[str]:
        """Extract text using PyMuPDF"""
        try:
            pdf_file.seek(0)
            pdf_bytes = pdf_file.read()
            
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            
            for page in doc:
                text += page.get_text() + "\n"
            
            doc.close()
            return text.strip() if len(text.strip()) >= TEXT_EXTRACTION_MIN_LENGTH else None
            
        except Exception:
            return None
    
    @staticmethod
    def extract_text_native(pdf_file) -> Tuple[Optional[str], str]:
        """
        Multi-stage native text extraction with fallbacks
        Returns: (extracted_text, method_used)
        """
        # Try PyPDF2 first
        text = PDFProcessor.extract_text_pypdf2(pdf_file)
        if text:
            return text, "pypdf2"
        
        # Try pdfplumber
        text = PDFProcessor.extract_text_pdfplumber(pdf_file)
        if text:
            return text, "pdfplumber"
        
        # Try PyMuPDF as last resort
        text = PDFProcessor.extract_text_pymupdf(pdf_file)
        if text:
            return text, "pymupdf"
        
        return None, "failed"
    
    @staticmethod
    def convert_pdf_to_images(pdf_file) -> List[Image.Image]:
        """Convert PDF pages to images for Vision API"""
        try:
            pdf_file.seek(0)
            pdf_bytes = pdf_file.read()
            
            images = convert_from_bytes(
                pdf_bytes,
                dpi=150,
                first_page=1,
                last_page=3  # Limit to first 3 pages for cost control
            )
            
            return images
            
        except Exception as e:
            st.error(f"Failed to convert PDF to images: {str(e)}")
            return []

class AIExtractor:
    """OpenAI API integration for text and vision processing"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def extract_text_from_images(self, images: List[Image.Image]) -> Optional[str]:
        """Use OpenAI Vision API to extract text from images"""
        try:
            all_text = ""
            
            for i, image in enumerate(images[:3]):  # Limit to 3 images
                # Convert PIL Image to base64
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_b64 = base64.b64encode(buffer.getvalue()).decode()
                
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL_VISION,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """Extract all text from this document image. 
                                    Return the text exactly as it appears, preserving formatting and structure.
                                    Focus on readable text and ignore decorative elements.
                                    If the image contains forms, preserve the relationship between labels and values."""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                
                page_text = response.choices[0].message.content
                all_text += f"\n--- Page {i+1} ---\n{page_text}\n"
            
            return all_text.strip() if len(all_text.strip()) >= TEXT_EXTRACTION_MIN_LENGTH else None
            
        except Exception as e:
            st.error(f"Vision API extraction failed: {str(e)}")
            return None
    
    def quick_scan_fields(self, text: str) -> List[str]:
        """Quick scan to identify all possible extractable fields"""
        try:
            prompt = f"""
Analyze this document text and identify all possible data fields that could be extracted.
Return only field names as a simple comma-separated list.
Examples: Customer Name, Policy Number, Claim Date, Amount, Branch Office
Focus on structured data fields that appear to have specific values.
Do not include generic text, paragraphs, or headers.
Look for form fields, labeled data, and key-value pairs.

Document text:
{text[:3000]}
"""
            
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL_TEXT,
                messages=[
                    {"role": "system", "content": "You are a document analysis assistant. Identify extractable data fields and return them as a comma-separated list."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse comma-separated fields
            fields = [field.strip() for field in response_text.split(',')]
            fields = [field for field in fields if field and len(field) > 2]
            
            return fields[:20]  # Limit to 20 fields
            
        except Exception as e:
            st.error(f"Quick scan failed: {str(e)}")
            return []
    
    def detailed_extraction(self, text: str, selected_fields: List[str]) -> Optional[Dict[str, Any]]:
        """Detailed extraction of selected fields only"""
        try:
            fields_str = ", ".join(selected_fields)
            
            prompt = f"""
Extract the following specific fields from this document: {fields_str}

Return as valid JSON with exact field_name: field_value pairs.
If a field is not found or unclear, return null for that field.
Extract values as plain text, preserve original formatting.
Be precise and only extract the actual value, not surrounding text or labels.

Expected format:
{{
  "Customer Name": "John Smith",
  "Policy Number": "POL-123456",
  "Missing Field": null
}}

Document text:
{text[:4000]}
"""
            
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL_TEXT,
                messages=[
                    {"role": "system", "content": "You are a document analysis assistant. Extract structured data from documents and return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_text = response_text[start_idx:end_idx]
                    extracted_data = json.loads(json_text)
                    
                    # Filter out null values
                    filtered_data = {k: v for k, v in extracted_data.items() if v is not None}
                    return filtered_data
                else:
                    return None
                    
            except json.JSONDecodeError:
                return None
                
        except Exception as e:
            st.error(f"Detailed extraction failed: {str(e)}")
            return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = []
    
    if 'discovered_fields' not in st.session_state:
        st.session_state.discovered_fields = []
    
    if 'selected_fields' not in st.session_state:
        st.session_state.selected_fields = []
    
    if 'processing_stage' not in st.session_state:
        st.session_state.processing_stage = "idle"

def create_download_link(data: Any, filename: str, format_type: str = "json") -> str:
    """Create download link for various formats"""
    if format_type == "json":
        content = json.dumps(data, indent=2, ensure_ascii=False)
        mime_type = "application/json"
    elif format_type == "csv":
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            content = df.to_csv(index=False)
        else:
            # Convert dict to single-row CSV
            df = pd.DataFrame([data])
            content = df.to_csv(index=False)
        mime_type = "text/csv"
    else:
        return ""
    
    b64 = base64.b64encode(content.encode()).decode()
    
    return f'''
    <a href="data:{mime_type};base64,{b64}" download="{filename}" 
       style="background-color: #4CAF50; color: white; padding: 8px 16px; 
              text-decoration: none; border-radius: 3px; display: inline-block; margin: 2px;">
        📥 {format_type.upper()}
    </a>
    '''

def process_single_pdf(pdf_file, api_key: str) -> Dict[str, Any]:
    """Process a single PDF through the complete pipeline"""
    result = {
        'filename': pdf_file.name,
        'status': 'failed',
        'error': '',
        'processing_method': '',
        'discovered_fields': [],
        'extracted_data': {},
        'job_id': None
    }
    
    try:
        # Calculate file hash
        pdf_file.seek(0)
        file_content = pdf_file.read()
        file_hash = PDFProcessor.get_file_hash(file_content)
        pdf_file.seek(0)
        
        # Save document to database
        doc = Document(
            filename=pdf_file.name,
            file_size=len(file_content),
            file_hash=file_hash,
            processing_method=""  # Will be updated
        )
        doc_id = st.session_state.db_manager.save_document(doc)
        
        # Stage 1: Native text extraction
        st.session_state.processing_stage = f"Extracting text from {pdf_file.name}..."
        text, method = PDFProcessor.extract_text_native(pdf_file)
        
        # Stage 2: OCR fallback if needed
        if not text:
            st.session_state.processing_stage = f"Converting {pdf_file.name} to images for OCR..."
            images = PDFProcessor.convert_pdf_to_images(pdf_file)
            
            if images:
                st.session_state.processing_stage = f"Performing OCR on {pdf_file.name}..."
                ai_extractor = AIExtractor(api_key)
                text = ai_extractor.extract_text_from_images(images)
                method = "ocr" if text else "failed"
            else:
                method = "failed"
        
        if not text:
            result['error'] = "Failed to extract text from PDF"
            return result
        
        # Update document with processing method
        doc.processing_method = method
        st.session_state.db_manager.save_document(doc)
        result['processing_method'] = method
        
        # Stage 3: AI field discovery
        st.session_state.processing_stage = f"Discovering fields in {pdf_file.name}..."
        ai_extractor = AIExtractor(api_key)
        discovered_fields = ai_extractor.quick_scan_fields(text)
        
        if not discovered_fields:
            result['error'] = "No extractable fields found"
            return result
        
        result['discovered_fields'] = discovered_fields
        result['status'] = 'fields_discovered'
        result['text_preview'] = text[:500] + "..." if len(text) > 500 else text
        
        # Create extraction job
        job = ExtractionJob(
            document_id=doc_id,
            status='pending',
            selected_fields=discovered_fields,
            total_fields_found=len(discovered_fields)
        )
        job_id = st.session_state.db_manager.save_extraction_job(job)
        result['job_id'] = job_id
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        st.error(f"Error processing {pdf_file.name}: {str(e)}")
        return result

def perform_detailed_extraction(job_id: int, selected_fields: List[str], text: str, api_key: str):
    """Perform detailed extraction for selected fields"""
    try:
        # Update job status
        st.session_state.db_manager.update_extraction_job(job_id, 'processing')
        
        # Extract selected fields
        ai_extractor = AIExtractor(api_key)
        extracted_data = ai_extractor.detailed_extraction(text, selected_fields)
        
        if extracted_data:
            # Save results
            st.session_state.db_manager.save_extracted_fields(job_id, extracted_data)
            st.session_state.db_manager.update_extraction_job(job_id, 'completed')
            return extracted_data
        else:
            st.session_state.db_manager.update_extraction_job(job_id, 'failed', 'No data extracted')
            return None
            
    except Exception as e:
        st.session_state.db_manager.update_extraction_job(job_id, 'failed', str(e))
        st.error(f"Detailed extraction failed: {str(e)}")
        return None

def render_field_selection_ui():
    """Render field selection interface"""
    if not st.session_state.discovered_fields:
        return
    
    st.subheader("🎯 Select Fields to Extract")
    
    # Get union of all discovered fields
    all_fields = []
    for result in st.session_state.processing_results:
        if 'discovered_fields' in result:
            all_fields.extend(result['discovered_fields'])
    
    # Remove duplicates while preserving order
    unique_fields = []
    seen = set()
    for field in all_fields:
        if field not in seen:
            unique_fields.append(field)
            seen.add(field)
    
    if not unique_fields:
        st.warning("No fields discovered in any document")
        return
    
    st.markdown('<div class="field-selection">', unsafe_allow_html=True)
    
    # Bulk selection controls
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("✅ Select All"):
            st.session_state.selected_fields = unique_fields.copy()
            st.experimental_rerun()
    
    with col2:
        if st.button("❌ Clear All"):
            st.session_state.selected_fields = []
            st.experimental_rerun()
    
    # Field checkboxes
    st.write(f"**Available Fields ({len(unique_fields)} found):**")
    
    # Initialize selected fields if empty
    if not st.session_state.selected_fields:
        st.session_state.selected_fields = unique_fields.copy()
    
    # Create checkboxes in columns
    num_cols = 3
    cols = st.columns(num_cols)
    
    for i, field in enumerate(unique_fields):
        col_idx = i % num_cols
        with cols[col_idx]:
            is_selected = field in st.session_state.selected_fields
            if st.checkbox(field, value=is_selected, key=f"field_{i}"):
                if field not in st.session_state.selected_fields:
                    st.session_state.selected_fields.append(field)
            else:
                if field in st.session_state.selected_fields:
                    st.session_state.selected_fields.remove(field)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show selected count
    st.info(f"📊 Selected {len(st.session_state.selected_fields)} out of {len(unique_fields)} fields")

def render_results_ui():
    """Render extraction results"""
    if not st.session_state.processing_results:
        return
    
    st.subheader("📊 Extraction Results")
    
    completed_results = []
    
    for result in st.session_state.processing_results:
        if 'extracted_data' in result and result['extracted_data']:
            completed_results.append(result)
    
    if not completed_results:
        st.info("No completed extractions yet. Process documents and select fields to see results.")
        return
    
    # Display results for each document
    for i, result in enumerate(completed_results):
        with st.expander(f"📄 {result['filename']} - {len(result['extracted_data'])} fields"):
            
            # Method and status info
            col1, col2 = st.columns(2)
            with col1:
                method_color = "🔤" if result['processing_method'] != 'ocr' else "👁️"
                st.write(f"**Method:** {method_color} {result['processing_method']}")
            with col2:
                st.write(f"**Status:** ✅ Completed")
            
            # Results table
            if result['extracted_data']:
                df = pd.DataFrame([
                    {"Field": k, "Value": v} 
                    for k, v in result['extracted_data'].items()
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Download buttons
                filename_base = result['filename'].replace('.pdf', '')
                json_link = create_download_link(result['extracted_data'], f"{filename_base}_data.json", "json")
                csv_link = create_download_link(result['extracted_data'], f"{filename_base}_data.csv", "csv")
                
                st.markdown(json_link + " " + csv_link, unsafe_allow_html=True)
    
    # Batch export if multiple results
    if len(completed_results) > 1:
        st.subheader("📦 Batch Export")
        
        # Combine all results
        batch_data = []
        for result in completed_results:
            row_data = {"filename": result['filename']}
            row_data.update(result['extracted_data'])
            batch_data.append(row_data)
        
        batch_json_link = create_download_link(batch_data, "batch_extraction_results.json", "json")
        batch_csv_link = create_download_link(batch_data, "batch_extraction_results.csv", "csv")
        
        st.markdown(batch_json_link + " " + batch_csv_link, unsafe_allow_html=True)

def render_history_ui():
    """Render processing history"""
    st.subheader("📈 Processing History")
    
    history = st.session_state.db_manager.get_extraction_history()
    
    if not history:
        st.info("No processing history yet.")
        return
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(history)
    df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Add download buttons for completed jobs
    for i, row in df.iterrows():
        if row['status'] == 'completed':
            job_results = st.session_state.db_manager.get_extraction_results(row['job_id'])
            if job_results:
                with st.expander(f"📄 {row['filename']} - {row['created_at']}"):
                    results_df = pd.DataFrame([
                        {"Field": k, "Value": v} 
                        for k, v in job_results.items()
                    ])
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    filename_base = row['filename'].replace('.pdf', '')
                    json_link = create_download_link(job_results, f"{filename_base}_historical.json", "json")
                    st.markdown(json_link, unsafe_allow_html=True)
    
    # Summary table
    st.dataframe(df[['filename', 'status', 'fields_found', 'method', 'created_at']], 
                use_container_width=True, hide_index=True)

def main():
    """Main Streamlit application"""
    
    # Initialize
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header"><h1>📄 PDF Parser Pro</h1></div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-powered PDF document analysis with batch processing and field extraction</p>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key for AI processing"
        )
        
        if not api_key:
            st.warning("⚠️ API key required")
        
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "📍 Navigate",
            ["🏠 Process Documents", "📊 View Results", "📈 History"],
            index=0
        )
        
        st.markdown("---")
        
        # Stats
        st.markdown("**📊 Quick Stats:**")
        history = st.session_state.db_manager.get_extraction_history(10)
        completed_jobs = len([h for h in history if h['status'] == 'completed'])
        st.metric("Completed Jobs", completed_jobs)
        
        if st.button("🗑️ Clear All Data"):
            if st.confirm("Delete all data?"):
                os.remove(DATABASE_PATH)
                st.session_state.db_manager = DatabaseManager()
                st.success("All data cleared!")
                st.experimental_rerun()
    
    # Main Content
    if page == "🏠 Process Documents":
        render_processing_page(api_key)
    elif page == "📊 View Results":
        render_results_ui()
    elif page == "📈 History":
        render_history_ui()

def render_processing_page(api_key: str):
    """Render the main processing page"""
    
    # Upload Section
    st.subheader("📁 Upload PDF Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help=f"Maximum {MAX_FILES_PER_UPLOAD} files, {MAX_FILE_SIZE // (1024*1024)}MB each"
    )
    
    if uploaded_files:
        # Validate files
        valid_files = []
        for file in uploaded_files[:MAX_FILES_PER_UPLOAD]:
            if file.size <= MAX_FILE_SIZE:
                valid_files.append(file)
            else:
                st.error(f"❌ {file.name} exceeds size limit")
        
        if valid_files:
            st.success(f"✅ {len(valid_files)} files ready for processing")
            
            # Process button
            if st.button("🚀 Process Documents", type="primary", disabled=not api_key):
                if not api_key:
                    st.error("Please enter your OpenAI API key first")
                    return
                
                # Initialize results
                st.session_state.processing_results = []
                st.session_state.discovered_fields = []
                
                # Process each file
                progress_bar = st.progress(0)
                status_container = st.empty()
                
                for i, pdf_file in enumerate(valid_files):
                    progress = (i + 1) / len(valid_files)
                    progress_bar.progress(progress)
                    
                    with status_container.container():
                        st.markdown(f'<div class="processing-stage">Processing {pdf_file.name}...</div>', 
                                  unsafe_allow_html=True)
                    
                    result = process_single_pdf(pdf_file, api_key)
                    st.session_state.processing_results.append(result)
                    
                    if result['status'] == 'fields_discovered':
                        st.session_state.discovered_fields.extend(result['discovered_fields'])
                
                status_container.success(f"✅ Processed {len(valid_files)} documents")
                st.experimental_rerun()
    
    # Field Selection
    if st.session_state.processing_results and any(r['status'] == 'fields_discovered' for r in st.session_state.processing_results):
        st.markdown("---")
        render_field_selection_ui()
        
        # Extract button
        if st.session_state.selected_fields and st.button("🎯 Extract Selected Fields", type="primary"):
            if not api_key:
                st.error("Please enter your OpenAI API key")
                return
            
            progress_bar = st.progress(0)
            status_container = st.empty()
            
            # Perform detailed extraction for each document
            extraction_count = 0
            for i, result in enumerate(st.session_state.processing_results):
                if result['status'] == 'fields_discovered' and 'job_id' in result:
                    
                    with status_container.container():
                        st.markdown(f'<div class="processing-stage">Extracting fields from {result["filename"]}...</div>', 
                                  unsafe_allow_html=True)
                    
                    extracted_data = perform_detailed_extraction(
                        result['job_id'], 
                        st.session_state.selected_fields,
                        result.get('text_preview', ''),
                        api_key
                    )
                    
                    if extracted_data:
                        result['extracted_data'] = extracted_data
                        result['status'] = 'completed'
                        extraction_count += 1
                    
                    progress = (i + 1) / len(st.session_state.processing_results)
                    progress_bar.progress(progress)
            
            status_container.success(f"✅ Completed extraction for {extraction_count} documents")
            st.experimental_rerun()
    
    # Show processing status
    if st.session_state.processing_stage != "idle":
        st.markdown(f'<div class="processing-stage">{st.session_state.processing_stage}</div>', 
                   unsafe_allow_html=True)
    
    # Quick results preview
    if st.session_state.processing_results:
        st.markdown("---")
        st.subheader("📋 Processing Summary")
        
        summary_data = []
        for result in st.session_state.processing_results:
            summary_data.append({
                "Filename": result['filename'],
                "Status": result['status'],
                "Method": result.get('processing_method', 'N/A'),
                "Fields Found": len(result.get('discovered_fields', [])),
                "Fields Extracted": len(result.get('extracted_data', {}))
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
