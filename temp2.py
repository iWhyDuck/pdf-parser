# app.py – Streamlit demo z dwoma trybami wyciągania danych z PDF-ów:
# 1) Classic (Regex)   2) AI (GPT-3.5 quick-scan → detailed)
# zależności: streamlit, pdfplumber, openai, sqlalchemy, concurrent.futures, langfuse

from __future__ import annotations
import io, json, os, re
from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from dotenv import load_dotenv
load_dotenv()

import pdfplumber
import streamlit as st
from openai import OpenAIError

# POPRAWIONE: Importy Langfuse dla v3
try:
    from langfuse.openai import OpenAI
    from langfuse import observe
except ImportError:
    # Fallback dla starszej wersji SDK (v2)
    try:
        from langfuse.openai import OpenAI
        from langfuse.decorators import observe
    except ImportError:
        st.error("❌ Nie można zaimportować Langfuse. Zainstaluj: pip install langfuse --upgrade")
        # Mock decorator jako fallback
        def observe(*args, **kwargs):
            def decorator(func):
                return func
            return decorator if args else decorator
        # Mock OpenAI client
        from openai import OpenAI

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool

import concurrent.futures

# ──────────────────────── 0. Custom Exceptions ────────────────────────
class PDFProcessingError(Exception):
    """Błąd podczas przetwarzania pliku PDF"""
    pass

class DataExtractionError(Exception):
    """Błąd podczas ekstrakcji danych"""
    pass

class DatabaseError(Exception):
    """Błąd podczas operacji na bazie danych"""
    pass

class ValidationError(Exception):
    """Błąd walidacji danych"""
    pass

# ──────────────────────── 1. Configuration ────────────────────────
class Config:
    OPENAI_MODEL = "gpt-3.5-turbo-1106"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MIN_FILE_SIZE = 100  # 100 bytes
    DATABASE_URL = "sqlite:///extractions.db"
    
    REGEX_FIELDS: Dict[str, Dict] = {
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
    __tablename__ = "extractions"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False)
    extraction_method = Column(String(50), nullable=False)
    extracted_data = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class DatabaseManager:
    """Zarządza połączeniami z bazą danych"""
    
    def __init__(self, database_url: str = Config.DATABASE_URL):
        self.database_url = database_url
        self._engine = None
        self._session_factory = None
    
    @property
    def engine(self):
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
                raise DatabaseError(f"Błąd inicjalizacji bazy danych: {str(e)}")
        return self._engine
    
    def create_session(self):
        """Tworzy nową sesję bazodanową"""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory()

class ExtractionRepository:
    """Repository pattern dla operacji na bazie danych"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def save_extraction(self, filename: str, file_hash: str, 
                       method: str, data: Dict[str, Any]) -> int:
        """Zapisuje wynik ekstrakcji do bazy"""
        session = self.db_manager.create_session()
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
            raise DatabaseError(f"Błąd zapisu do bazy: {str(e)}")
        finally:
            session.close()

# ──────────────────────── 3. Validation ────────────────────────
class PDFValidator:
    """Waliduje pliki PDF pod kątem bezpieczeństwa i poprawności"""
    
    @staticmethod
    def validate_pdf_file(pdf_bytes: bytes, filename: str) -> None:
        PDFValidator._validate_file_size(pdf_bytes, filename)
        PDFValidator._validate_pdf_format(pdf_bytes, filename)
        PDFValidator._validate_file_extension(filename)
    
    @staticmethod
    def _validate_file_size(pdf_bytes: bytes, filename: str) -> None:
        if len(pdf_bytes) > Config.MAX_FILE_SIZE:
            raise ValidationError(f"Plik {filename} jest za duży. Maksymalny rozmiar: {Config.MAX_FILE_SIZE // (1024*1024)}MB")
        
        if len(pdf_bytes) < Config.MIN_FILE_SIZE:
            raise ValidationError(f"Plik {filename} jest za mały lub uszkodzony")
    
    @staticmethod
    def _validate_pdf_format(pdf_bytes: bytes, filename: str) -> None:
        if not pdf_bytes.startswith(b'%PDF'):
            raise ValidationError(f"Plik {filename} nie jest prawidłowym plikiem PDF")
    
    @staticmethod
    def _validate_file_extension(filename: str) -> None:
        if not filename.lower().endswith('.pdf'):
            raise ValidationError(f"Nieprawidłowe rozszerzenie pliku. Oczekiwano .pdf, otrzymano: {Path(filename).suffix}")

# ──────────────────────── 4. Text Extraction ────────────────────────
class TextExtractor:
    """Ekstraktuje tekst z plików PDF"""
    
    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                if not pdf.pages:
                    raise PDFProcessingError("PDF nie zawiera żadnych stron")
                
                text_parts = []
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        st.warning(f"⚠️ Nie udało się przetworzyć strony {i+1}: {str(e)}")
                        continue
                
                if not text_parts:
                    raise PDFProcessingError("Nie udało się wyekstraktować tekstu z żadnej strony PDF")
                
                return "\n".join(text_parts)
                
        except Exception as e:
            if isinstance(e, PDFProcessingError):
                raise
            raise PDFProcessingError(f"Błąd podczas czytania PDF: {str(e)}")

# ──────────────────────── 5. Data Extractors ────────────────────────
class DataExtractor(ABC):
    """Abstrakcyjna klasa bazowa dla ekstraktowania danych"""
    
    @abstractmethod
    def extract(self, text: str, fields: List[str]) -> Dict[str, str]:
        pass

class ClassicExtractor(DataExtractor):
    """Ekstraktuje dane używając wyrażeń regularnych"""
    
    def __init__(self, config: Dict[str, Dict] = Config.REGEX_FIELDS):
        try:
            self.cfg = {
                k: [re.compile(p, re.I) for p in v["patterns"]] 
                for k, v in config.items()
            }
        except re.error as e:
            raise DataExtractionError(f"Błąd w wyrażeniu regularnym: {str(e)}")

    @observe(name="classic_extraction")
    def extract(self, text: str, fields: List[str] = None) -> Dict[str, str]:
        if not text or not text.strip():
            raise DataExtractionError("Brak tekstu do przetworzenia")
        
        try:
            out: Dict[str, str] = {}
            fields_to_extract = fields if fields else self.cfg.keys()
            
            for key in fields_to_extract:
                if key not in self.cfg:
                    st.warning(f"⚠️ Nieznane pole: {key}")
                    continue
                
                value = self._extract_field_value(key, text)
                if value:
                    out[key] = value
            
            return out
            
        except Exception as e:
            raise DataExtractionError(f"Błąd podczas ekstrakcji klasycznej: {str(e)}")
    
    def _extract_field_value(self, key: str, text: str) -> Optional[str]:
        """Ekstraktuje wartość dla konkretnego pola"""
        for pat in self.cfg[key]:
            try:
                if m := pat.search(text):
                    return m.group(1).strip()
            except Exception as e:
                st.warning(f"⚠️ Błąd podczas przetwarzania pola '{key}': {str(e)}")
                continue
        return None

class AIExtractor(DataExtractor):
    """Ekstraktuje dane używając AI/GPT"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise DataExtractionError("Brak klucza API OpenAI")
        
        try:
            self.cli = OpenAI(api_key=api_key)
        except Exception as e:
            raise DataExtractionError(f"Błąd inicjalizacji klienta OpenAI: {str(e)}")

    @observe(name="openai_chat_completion", as_type="generation")
    def _chat(self, messages: List[Dict], **kw) -> str:
        try:
            r = self.cli.chat.completions.create(
                model=Config.OPENAI_MODEL, messages=messages, temperature=0, **kw
            )
            if not r.choices or not r.choices[0].message.content:
                raise DataExtractionError("OpenAI zwróciło pustą odpowiedź")
            
            return r.choices[0].message.content.strip()
            
        except OpenAIError as e:
            raise DataExtractionError(f"Błąd API OpenAI: {str(e)}")
        except Exception as e:
            raise DataExtractionError(f"Nieoczekiwany błąd podczas wywołania AI: {str(e)}")

    @observe(name="discover_labels")
    def discover_labels(self, text: str, max_labels: int = 15) -> List[str]:
        if not text or not text.strip():
            raise DataExtractionError("Brak tekstu do analizy")
        
        try:
            prompt = (
                "Return comma-separated labels (no values) that look like form-field names "
                f"in the document below (≤{max_labels}).\n\n{text[:3000]}"
            )
            raw = self._chat(
                [
                    {"role": "system", "content": "You are PDF-data assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
            )
            
            if not raw:
                return []
            
            labels = [l.strip() for l in raw.split(",") if 2 < len(l.strip()) < 40]
            return labels[:max_labels]
            
        except Exception as e:
            if isinstance(e, DataExtractionError):
                raise
            raise DataExtractionError(f"Błąd podczas odkrywania etykiet: {str(e)}")

    @observe(name="ai_extract_data")
    def extract(self, text: str, fields: List[str]) -> Dict[str, str]:
        if not text or not text.strip():
            raise DataExtractionError("Brak tekstu do przetworzenia")
        
        if not fields:
            raise DataExtractionError("Brak pól do ekstrakcji")
        
        try:
            prompt = self._build_extraction_prompt(fields, text)
            raw = self._chat(
                [
                    {"role": "system", "content": "You are data-extraction engine."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
            )
            
            return self._parse_extraction_result(raw)
                
        except Exception as e:
            if isinstance(e, DataExtractionError):
                raise
            raise DataExtractionError(f"Błąd podczas ekstrakcji AI: {str(e)}")
    
    def _build_extraction_prompt(self, fields: List[str], text: str) -> str:
        return (
            f"Extract: {', '.join(fields)}\n\n"
            "Return ONLY compact JSON {\"Field\":\"Value\"}. "
            "If a field is missing, set null.\n\n"
            + text[:20_000]
        )
    
    def _parse_extraction_result(self, raw_response: str) -> Dict[str, str]:
        m = re.search(r"\{.*\}", raw_response, re.S)
        if not m:
            raise DataExtractionError("AI nie zwróciło prawidłowego JSON-a")
        
        try:
            result = json.loads(m.group(0))
            if not isinstance(result, dict):
                raise DataExtractionError("AI zwróciło nieprawidłowy format danych")
            
            return result
            
        except json.JSONDecodeError as e:
            raise DataExtractionError(f"Błąd parsowania JSON z AI: {str(e)}")

# ──────────────────────── 6. PDF Processing Service ────────────────────────
class PDFProcessor:
    """Główna klasa do przetwarzania plików PDF"""
    
    def __init__(self, repository: ExtractionRepository):
        self.repository = repository
        self.text_extractor = TextExtractor()
        self.validator = PDFValidator()
    
    def process_file(self, file_bytes: bytes, filename: str, 
                    extractor: DataExtractor, fields: List[str]) -> Tuple[Dict[str, str], str]:
        """Przetwarza pojedynczy plik PDF"""
        self.validator.validate_pdf_file(file_bytes, filename)
        text = self.text_extractor.extract_text(file_bytes)
        file_hash = sha256(file_bytes).hexdigest()[:6]
        
        data = extractor.extract(text, fields)
        
        return data, file_hash
    
    def save_extraction_result(self, filename: str, file_hash: str, 
                              method: str, data: Dict[str, str]) -> int:
        """Zapisuje wynik ekstrakcji"""
        return self.repository.save_extraction(filename, file_hash, method, data)

class BatchProcessor:
    """Przetwarza wiele plików PDF jednocześnie"""
    
    def __init__(self, pdf_processor: PDFProcessor):
        self.pdf_processor = pdf_processor
    
    @observe(name="batch_pdf_processing")
    def process_batch(self, uploaded_files, extractor: DataExtractor, 
                     fields: List[str], method: str) -> List[Dict]:
        batch_results = []
        
        for up in uploaded_files:
            try:
                pdf_bytes = up.read()
                data, file_hash = self.pdf_processor.process_file(
                    pdf_bytes, up.name, extractor, fields
                )
                
                db_id = self.pdf_processor.save_extraction_result(
                    up.name, file_hash, method, data
                )
                
                batch_results.append({
                    "file": up.name,
                    "result": data,
                    "db_id": db_id
                })
                st.write(f"✅ Przetworzono: {up.name}")
                
            except Exception as e:
                batch_results.append({
                    "file": up.name,
                    "error": str(e)
                })
                st.write(f"❌ Błąd w {up.name}: {str(e)}")
        
        return batch_results

# ──────────────────────── 7. UI Components ────────────────────────
class FieldSelector:
    """Komponent do wyboru pól do ekstrakcji"""
    
    @staticmethod
    def render_field_checkboxes(fields_config: Dict[str, Dict], key_prefix: str) -> List[str]:
        """Renderuje checkboxy dla pól i zwraca wybrane"""
        cols = st.columns(3)
        selected_fields = []
        field_keys = list(fields_config.keys())
        
        for i, field_key in enumerate(field_keys):
            display_name = fields_config[field_key]['display']
            if cols[i % 3].checkbox(display_name, True, key=f"{key_prefix}_{field_key}"):
                selected_fields.append(field_key)
        
        return selected_fields
    
    @staticmethod
    def render_ai_field_checkboxes(labels: List[str], key_prefix: str) -> List[str]:
        """Renderuje checkboxy dla pól AI i zwraca wybrane"""
        cols = st.columns(3)
        selected_fields = []
        
        for i, label in enumerate(labels):
            if cols[i % 3].checkbox(label, True, key=f"{key_prefix}_{label}"):
                selected_fields.append(label)
        
        return selected_fields

class UIRenderer:
    """Renderuje interfejs użytkownika"""
    
    def __init__(self):
        self.field_selector = FieldSelector()
    
    def render_header(self):
        st.set_page_config(page_title="PDF Extractor", layout="wide")
        st.title("📄 PDF Extractor — Classic vs AI")
        
        if self._check_langfuse_config():
            st.info("🔍 Langfuse monitoring aktywny")
        else:
            st.info("📊 Monitoring wyłączony - dodaj LANGFUSE_PUBLIC_KEY i LANGFUSE_SECRET_KEY")
    
    def render_mode_selector(self) -> str:
        return st.radio("Tryb parsera:", ["Classic (Regex)", "AI (GPT-3.5)"], horizontal=True)
    
    def render_batch_section(self) -> Tuple[List, List[str]]:
        st.header("📦 Batch PDF Extraction (asynchroniczne zapisywanie do bazy)")
        
        uploaded_files = st.file_uploader(
            "Wgraj wiele plików PDF",
            type="pdf",
            accept_multiple_files=True,
            key="batch_uploader"
        )
        
        selected_fields = []
        if uploaded_files:
            st.subheader("🔤 Dostępne pola do batch extraction")
            selected_fields = self.field_selector.render_field_checkboxes(
                Config.REGEX_FIELDS, "batch"
            )
        
        return uploaded_files, selected_fields
    
    def render_single_file_uploader(self):
        return st.file_uploader("Wgraj PDF", type="pdf", key="single_uploader")
    
    def render_extraction_results(self, data: Dict[str, str], filename: str, 
                                file_hash: str, db_id: int):
        """Renderuje wyniki ekstrakcji"""
        st.success("Ekstrakcja zakończona")
        
        for k, v in data.items():
            display_name = Config.REGEX_FIELDS.get(k, {}).get('display', k)
            st.write(f"**{display_name}**: {v}")
        
        st.success(f"✅ Dane zapisane do bazy (ID: {db_id})")
        
        st.download_button(
            "💾 Pobierz JSON",
            json.dumps(data, ensure_ascii=False, indent=2),
            file_name=f"{Path(filename).stem}_{file_hash}.json",
            mime="application/json",
        )
    
    def _check_langfuse_config(self) -> bool:
        required_keys = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        return len(missing_keys) == 0

# ──────────────────────── 8. Main Application ────────────────────────
class PDFExtractorApp:
    """Główna aplikacja Streamlit"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.repository = ExtractionRepository(self.db_manager)
        self.processor = PDFProcessor(self.repository)
        self.batch_processor = BatchProcessor(self.processor)
        self.ui = UIRenderer()
    
    def run(self):
        """Uruchamia aplikację"""
        self.ui.render_header()
        
        mode = self.ui.render_mode_selector()
        
        if mode.startswith("AI") and not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ Ustaw zmienną OPENAI_API_KEY lub dodaj ją w .secrets.toml")
            return
        
        # Batch processing section
        self._handle_batch_processing(mode)
        
        # Single file processing section
        self._handle_single_file_processing(mode)
    
    def _handle_batch_processing(self, mode: str):
        """Obsługuje przetwarzanie wielu plików"""
        uploaded_files, selected_fields = self.ui.render_batch_section()
        
        if uploaded_files and selected_fields:
            if st.button("Extract All (async batch & save to DB)"):
                try:
                    extractor = self._create_extractor(mode)
                    method = "classic" if mode.startswith("Classic") else "ai"
                    
                    batch_results = self.batch_processor.process_batch(
                        uploaded_files, extractor, selected_fields, method
                    )
                    
                    self._display_batch_results(batch_results)
                    
                except Exception as e:
                    st.error(f"❌ Błąd batch processing: {str(e)}")
    
    def _handle_single_file_processing(self, mode: str):
        """Obsługuje przetwarzanie pojedynczego pliku"""
        uploaded_file = self.ui.render_single_file_uploader()
        if not uploaded_file:
            return
        
        try:
            if mode.startswith("Classic"):
                self._handle_classic_mode(uploaded_file)
            else:
                self._handle_ai_mode(uploaded_file)
                
        except Exception as e:
            st.error(f"❌ Błąd podczas przetwarzania: {str(e)}")
    
    def _handle_classic_mode(self, uploaded_file):
        """Obsługuje tryb klasyczny"""
        st.header("🔤 Dostępne pola (regex)")
        
        selected_fields = self.ui.field_selector.render_field_checkboxes(
            Config.REGEX_FIELDS, "classic"
        )
        
        if st.button("Extract selected fields", key="single_extract_button"):
            self._process_single_file(uploaded_file, "Classic", selected_fields)
    
    def _handle_ai_mode(self, uploaded_file):
        """Obsługuje tryb AI"""
        pdf_bytes = uploaded_file.read()
        PDFValidator.validate_pdf_file(pdf_bytes, uploaded_file.name)
        text = TextExtractor.extract_text(pdf_bytes)
        
        extractor = AIExtractor(os.getenv("OPENAI_API_KEY"))
        st.header("🤖 AI Quick-scan")
        
        with st.spinner("Skanowanie dokumentu..."):
            labels = extractor.discover_labels(text)
        
        if not labels:
            st.warning("Model nie zidentyfikował etykiet.")
            return
        
        selected_fields = self.ui.field_selector.render_ai_field_checkboxes(
            labels, "ai_field"
        )
        
        if st.button("Extract selected fields", key="single_ai_extract_button"):
            try:
                file_hash = sha256(pdf_bytes).hexdigest()[:6]
                data = extractor.extract(text, selected_fields)
                
                if data:
                    db_id = self.processor.save_extraction_result(
                        uploaded_file.name, file_hash, "ai", data
                    )
                    self.ui.render_extraction_results(
                        data, uploaded_file.name, file_hash, db_id
                    )
                else:
                    st.warning("Nie udało się wyekstraktować żadnych danych.")
                    
            except Exception as e:
                st.error(f"❌ Błąd podczas ekstrakcji AI: {str(e)}")
    
    def _process_single_file(self, uploaded_file, mode: str, selected_fields: List[str]):
        """Przetwarza pojedynczy plik"""
        if not selected_fields:
            st.warning("Wybierz co najmniej jedno pole do ekstrakcji.")
            return
        
        try:
            with st.spinner("Ekstraktowanie danych..."):
                pdf_bytes = uploaded_file.read()
                extractor = self._create_extractor(mode)
                
                data, file_hash = self.processor.process_file(
                    pdf_bytes, uploaded_file.name, extractor, selected_fields
                )
                
                if data:
                    method = "classic" if mode.startswith("Classic") else "ai"
                    db_id = self.processor.save_extraction_result(
                        uploaded_file.name, file_hash, method, data
                    )
                    self.ui.render_extraction_results(
                        data, uploaded_file.name, file_hash, db_id
                    )
                else:
                    st.warning("Żadne z wybranych pól nie zostało znalezione.")
                    
        except Exception as e:
            st.error(f"❌ Błąd podczas ekstrakcji: {str(e)}")
    
    def _create_extractor(self, mode: str) -> DataExtractor:
        """Factory method dla tworzenia extractorów"""
        if mode.startswith("Classic"):
            return ClassicExtractor()
        else:
            return AIExtractor(os.getenv("OPENAI_API_KEY"))
    
    def _display_batch_results(self, batch_results: List[Dict]):
        """Wyświetla wyniki batch processing"""
        st.success("✅ Wszystkie pliki przetworzone i zapisane!")
        st.subheader("Wyniki batch extraction")
        
        for br in batch_results:
            st.write(f"**Plik:** {br['file']}")
            if "result" in br:
                st.json(br["result"])
                st.write(f"Zapisano w bazie (ID: {br['db_id']})")
            else:
                st.error(f"Błąd: {br['error']}")
        
        st.download_button(
            "💾 Pobierz batch JSON",
            json.dumps(batch_results, ensure_ascii=False, indent=2),
            file_name="batch_results.json",
            mime="application/json",
        )

# ──────────────────────── 9. Entry Point ────────────────────────
if __name__ == "__main__":
    try:
        app = PDFExtractorApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Krytyczny błąd aplikacji: {str(e)}")
