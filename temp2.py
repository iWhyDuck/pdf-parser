# app.py – Streamlit demo z dwoma trybami wyciągania danych z PDF-ów:
# 1) Classic (Regex)   2) AI (GPT-3.5 quick-scan → detailed)
# zależności: streamlit, pdfplumber, openai, sqlalchemy, concurrent.futures, langfuse

from __future__ import annotations
import io, json, os, re
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from dotenv import load_dotenv
load_dotenv()

import pdfplumber
import streamlit as st
from langfuse.openai import OpenAI
from openai import OpenAIError
from langfuse import observe  # POPRAWIONE: import dla SDK v3

# Dodane importy dla bazy danych
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, func
from sqlalchemy.orm import declarative_base
Base = declarative_base()
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool  # DODANE: dla lepszego poolingu SQLite

import concurrent.futures  # DODANE: dla batch processing

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

# ──────────────────────── 1. Walidacja i bezpieczeństwo ────────────────────────
class PDFValidator:
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MIN_FILE_SIZE = 100  # 100 bytes
    
    @staticmethod
    def validate_pdf_file(pdf_bytes: bytes, filename: str) -> None:
        """Waliduje plik PDF pod kątem bezpieczeństwa i poprawności"""
        
        # Sprawdź rozmiar pliku
        if len(pdf_bytes) > PDFValidator.MAX_FILE_SIZE:
            raise ValidationError(f"Plik {filename} jest za duży. Maksymalny rozmiar: {PDFValidator.MAX_FILE_SIZE // (1024*1024)}MB")
        
        if len(pdf_bytes) < PDFValidator.MIN_FILE_SIZE:
            raise ValidationError(f"Plik {filename} jest za mały lub uszkodzony")
        
        # Sprawdź czy to rzeczywiście PDF
        if not pdf_bytes.startswith(b'%PDF'):
            raise ValidationError(f"Plik {filename} nie jest prawidłowym plikiem PDF")
        
        # Sprawdź rozszerzenie pliku
        if not filename.lower().endswith('.pdf'):
            raise ValidationError(f"Nieprawidłowe rozszerzenie pliku. Oczekiwano .pdf, otrzymano: {Path(filename).suffix}")

# ──────────────────────── 2. Konfiguracja ────────────────────────
OPENAI_MODEL = "gpt-3.5-turbo-1106"        # 4o można też, ale drożej
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

# ──────────────────────── 3. Database Models ────────────────────────
Base = declarative_base()

class Extraction(Base):
    __tablename__ = "extractions"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False)
    extraction_method = Column(String(50), nullable=False)
    extracted_data = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

@st.cache_resource
def get_db_session():
    try:
        engine = create_engine("sqlite:///extractions.db", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        return sessionmaker(bind=engine)()
    except Exception as e:
        raise DatabaseError(f"Błąd inicjalizacji bazy danych: {str(e)}")

# POPRAWIONE: Dla operacji wielowątkowych z lepszą konfiguracją SQLite
def create_db_session():
    """Tworzy nową sesję dla każdego wywołania - bezpieczne dla wielu wątków"""
    try:
        engine = create_engine(
            "sqlite:///extractions.db", 
            connect_args={
                "check_same_thread": False,
                "timeout": 20  # Timeout dla SQLite
            },
            poolclass=StaticPool,  # StaticPool dla SQLite
            echo=False  # Ustaw na True żeby zobaczyć SQL queries w logach
        )
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        return Session()
    except Exception as e:
        raise DatabaseError(f"Błąd inicjalizacji bazy danych: {str(e)}")

# ──────────────────────── 4. Extractor classic ─────────────────────
class ClassicExtractor:
    def __init__(self, config: Dict[str, Dict]):
        try:
            self.cfg = {
                k: [re.compile(p, re.I) for p in v["patterns"]] for k, v in config.items()
            }
        except re.error as e:
            raise DataExtractionError(f"Błąd w wyrażeniu regularnym: {str(e)}")

    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        """Ekstraktuje tekst z PDF z obsługą błędów"""
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

    @observe(name="classic_extraction")
    def run(self, text: str, selected_fields: List[str] = None) -> Dict[str, str]:
        """Uruchamia ekstrakcję z obsługą błędów"""
        if not text or not text.strip():
            raise DataExtractionError("Brak tekstu do przetworzenia")
        
        try:
            out: Dict[str, str] = {}
            fields_to_extract = selected_fields if selected_fields else self.cfg.keys()
            
            for key in fields_to_extract:
                if key not in self.cfg:
                    st.warning(f"⚠️ Nieznane pole: {key}")
                    continue
                    
                for pat in self.cfg[key]:
                    try:
                        if m := pat.search(text):
                            out[key] = m.group(1).strip()
                            break
                    except Exception as e:
                        st.warning(f"⚠️ Błąd podczas przetwarzania pola '{key}': {str(e)}")
                        continue
            
            return out
            
        except Exception as e:
            raise DataExtractionError(f"Błąd podczas ekstrakcji klasycznej: {str(e)}")


# ──────────────────────── 5. Extractor AI ─────────────────────────
class AIExtractor:
    def __init__(self, api_key: str):
        if not api_key:
            raise DataExtractionError("Brak klucza API OpenAI")
        
        try:
            self.cli = OpenAI(api_key=api_key)
        except Exception as e:
            raise DataExtractionError(f"Błąd inicjalizacji klienta OpenAI: {str(e)}")

    @observe(name="openai_chat_completion", as_type="generation")
    def _chat(self, messages: List[Dict], **kw) -> str:
        """Wywołanie API z obsługą błędów"""
        try:
            r = self.cli.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, temperature=0, **kw
            )
            if not r.choices or not r.choices[0].message.content:
                raise DataExtractionError("OpenAI zwróciło pustą odpowiedź")
            
            return r.choices[0].message.content.strip()
            
        except OpenAIError as e:
            raise DataExtractionError(f"Błąd API OpenAI: {str(e)}")
        except Exception as e:
            raise DataExtractionError(f"Nieoczekiwany błąd podczas wywołania AI: {str(e)}")

    @observe(name="discover_labels")
    def discover(self, text: str, max_labels: int = 15) -> List[str]:
        """Odkrywa etykiety z obsługą błędów"""
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
        """Ekstraktuje dane z obsługą błędów"""
        if not text or not text.strip():
            raise DataExtractionError("Brak tekstu do przetworzenia")
        
        if not fields:
            raise DataExtractionError("Brak pól do ekstrakcji")
        
        try:
            prompt = (
                f"Extract: {', '.join(fields)}\n\n"
                "Return ONLY compact JSON {\"Field\":\"Value\"}. "
                "If a field is missing, set null.\n\n"
                + text[:20_000]
            )
            raw = self._chat(
                [
                    {"role": "system", "content": "You are data-extraction engine."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
            )
            
            # Znajdź JSON w odpowiedzi
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                raise DataExtractionError("AI nie zwróciło prawidłowego JSON-a")
            
            try:
                result = json.loads(m.group(0))
                if not isinstance(result, dict):
                    raise DataExtractionError("AI zwróciło nieprawidłowy format danych")
                
                return result
                
            except json.JSONDecodeError as e:
                raise DataExtractionError(f"Błąd parsowania JSON z AI: {str(e)}")
                
        except Exception as e:
            if isinstance(e, DataExtractionError):
                raise
            raise DataExtractionError(f"Błąd podczas ekstrakcji AI: {str(e)}")


# ──────────────────────── 6. UI Streamlit ─────────────────────────
st.set_page_config(page_title="PDF Extractor", layout="wide")
st.title("📄 PDF Extractor — Classic vs AI")

# Sprawdzenie konfiguracji Langfuse
def check_langfuse_config():
    required_keys = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    return len(missing_keys) == 0

# Langfuse monitoring info
if check_langfuse_config():
    st.info("🔍 Langfuse monitoring aktywny")
else:
    st.info("📊 Monitoring wyłączony - dodaj LANGFUSE_PUBLIC_KEY i LANGFUSE_SECRET_KEY")

mode = st.radio("Tryb parsera:", ["Classic (Regex)", "AI (GPT-3.5)"], horizontal=True)

if mode.startswith("AI") and not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Ustaw zmienną OPENAI_API_KEY lub dodaj ją w .secrets.toml")
    st.stop()

# ──────────────────────── FUNKCJE WYSOKIEGO POZIOMU Z DEKORATORAMI ────────────────────────────

@observe(name="batch_pdf_processing")
def process_batch_pdfs(uploaded_files, mode: str, selected_fields: List[str]):
    """Batch processing PDF files z pełnym monitoringiem Langfuse"""
    batch_results = []
    
    # Jedna sesja dla całego batcha
    session = create_db_session()
    
    try:
        for up in uploaded_files:
            try:
                pdf_bytes = up.read()
                PDFValidator.validate_pdf_file(pdf_bytes, up.name)
                text = ClassicExtractor.extract_text(pdf_bytes)
                file_hash = sha256(pdf_bytes).hexdigest()[:6]
                
                if mode.startswith("Classic"):
                    extractor = ClassicExtractor(REGEX_FIELDS)
                    data = extractor.run(text, selected_fields)
                else:
                    extractor = AIExtractor(os.getenv("OPENAI_API_KEY"))
                    data = extractor.extract(text, selected_fields)
                
                record = Extraction(
                    filename=up.name,
                    file_hash=file_hash,
                    extraction_method= "classic" if mode.startswith("Classic") else "ai",
                    extracted_data=json.dumps(data, ensure_ascii=False)
                )
                session.add(record)
                session.flush()  # Flush bez commit
                
                batch_results.append({
                    "file": up.name,
                    "result": data,
                    "db_id": record.id
                })
                st.write(f"✅ Przetworzono: {up.name}")  # Debug info
                
            except Exception as e:
                batch_results.append({
                    "file": up.name,
                    "error": str(e)
                })
                st.write(f"❌ Błąd w {up.name}: {str(e)}")  # Debug info
        
        # Jeden commit na koniec
        session.commit()
        st.success("✅ Wszystkie pliki przetworzone i zapisane!")
        
    except Exception as e:
        session.rollback()
        st.error(f"❌ Błąd batch processing: {str(e)}")
        raise e
    finally:
        session.close()

    return batch_results

@observe(name="single_pdf_processing")
def process_single_pdf(uploaded_file, mode: str, selected_fields: List[str]):
    """Single PDF processing z pełnym monitoringiem Langfuse"""
    try:
        # Wczytaj i zwaliduj plik
        pdf_bytes = uploaded_file.read()
        PDFValidator.validate_pdf_file(pdf_bytes, uploaded_file.name)
        
        # Ekstraktuj tekst
        text = ClassicExtractor.extract_text(pdf_bytes)
        file_hash = sha256(pdf_bytes).hexdigest()[:6]
        
        # Przetwarzanie w zależności od trybu
        if mode.startswith("Classic"):
            extractor = ClassicExtractor(REGEX_FIELDS)
            data = extractor.run(text, selected_fields)
        else:
            extractor = AIExtractor(os.getenv("OPENAI_API_KEY"))
            data = extractor.extract(text, selected_fields)
        
        return data, file_hash, text
        
    except Exception as e:
        raise e

# ──────────────────────── *** BATCH PDF SECTION *** ────────────────────────────
st.header("📦 Batch PDF Extraction (asynchroniczne zapisywanie do bazy)")

uploaded_files = st.file_uploader(
    "Wgraj wiele plików PDF",
    type="pdf",
    accept_multiple_files=True,
    key="batch_uploader"
)

if uploaded_files:
    st.subheader("🔤 Dostępne pola do batch extraction")
    batch_field_keys = list(REGEX_FIELDS.keys())
    batch_cols = st.columns(3)
    batch_selected_fields: List[str] = []
    for i, field_key in enumerate(batch_field_keys):
        display_name = REGEX_FIELDS[field_key]['display']
        if batch_cols[i % 3].checkbox(display_name, True, key=f"batch_{field_key}"):
            batch_selected_fields.append(field_key)

    if not batch_selected_fields:
        st.warning("Wybierz co najmniej jedno pole do ekstrakcji.")
    
    if st.button("Extract All (async batch & save to DB)"):
        batch_results = process_batch_pdfs(uploaded_files, mode, batch_selected_fields)
        
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

# ──────────────────────── 7. Główny proces z error handling ─────────────────────
# Pojedynczy plik
up = st.file_uploader("Wgraj PDF", type="pdf", key="single_uploader")
if not up:
    st.stop()

try:
    # Inicjalizacja sesji bazy danych
    session = get_db_session()
except DatabaseError as e:
    st.error(f"❌ {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"❌ Nieoczekiwany błąd inicjalizacji bazy: {str(e)}")
    st.stop()

# ───────────────────── Classic flow ───────────────────────────────
if mode.startswith("Classic"):
    st.header("🔤 Dostępne pola (regex)")
    
    # Pokaż checkboxy dla dostępnych pól
    cols = st.columns(3)
    selected: List[str] = []
    field_keys = list(REGEX_FIELDS.keys())
    
    for i, field_key in enumerate(field_keys):
        display_name = REGEX_FIELDS[field_key]['display']
        if cols[i % 3].checkbox(display_name, True, key=f"classic_{field_key}"):
            selected.append(field_key)

    if st.button("Extract selected fields", key="single_extract_button"):
        try:
            with st.spinner("Ekstraktowanie danych..."):
                data, file_hash, _ = process_single_pdf(up, mode, selected)
                
                if data:
                    st.success("Ekstrakcja zakończona")
                    for k, v in data.items():
                        display_name = REGEX_FIELDS.get(k, {}).get('display', k)
                        st.write(f"**{display_name}**: {v}")
                    
                    # Zapis do bazy danych z obsługą błędów
                    try:
                        record = Extraction(
                            filename=up.name,
                            file_hash=file_hash,
                            extraction_method="classic",
                            extracted_data=json.dumps(data, ensure_ascii=False)
                        )
                        session.add(record)
                        session.commit()
                        st.success(f"✅ Dane zapisane do bazy (ID: {record.id})")
                    except SQLAlchemyError as e:
                        st.error(f"❌ Błąd zapisu do bazy: {str(e)}")
                        session.rollback()
                    
                    # Pobieranie JSON
                    st.download_button(
                        "💾 Pobierz JSON",
                        json.dumps(data, ensure_ascii=False, indent=2),
                        file_name=f"{Path(up.name).stem}_{file_hash}.json",
                        mime="application/json",
                    )
                else:
                    st.warning("Żadne z wybranych pól nie zostało znalezione.")
                    
        except DataExtractionError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ Nieoczekiwany błąd podczas ekstrakcji: {str(e)}")

# ───────────────────── AI flow ────────────────────────────────────
else:
    try:
        # Wczesne wczytanie pliku dla AI flow
        pdf_bytes = up.read()
        PDFValidator.validate_pdf_file(pdf_bytes, up.name)
        text = ClassicExtractor.extract_text(pdf_bytes)
        
        api = AIExtractor(os.getenv("OPENAI_API_KEY"))
        st.header("🤖 AI Quick-scan")

        with st.spinner("Skanowanie dokumentu..."):
            labels = api.discover(text)
            
        if not labels:
            st.warning("Model nie zidentyfikował etykiet.")
            st.stop()

        # checkboxy wyboru etykiet
        cols = st.columns(3)
        selected: List[str] = []
        for i, lab in enumerate(labels):
            if cols[i % 3].checkbox(lab, True, key=f"ai_field_{lab}"):
                selected.append(lab)

        if st.button("Extract selected fields", key="single_ai_extract_button"):
            try:
                with st.spinner("Ekstraktowanie danych przez AI..."):
                    # Użyj już wczytanego tekstu zamiast ponownego odczytu pliku
                    file_hash = sha256(pdf_bytes).hexdigest()[:6]
                    data = api.extract(text, selected)
                    
                    if data:
                        st.success("Ekstrakcja zakończona")
                        st.json(data)
                        
                        # Zapis do bazy danych z obsługą błędów
                        try:
                            record = Extraction(
                                filename=up.name,
                                file_hash=file_hash,
                                extraction_method="ai",
                                extracted_data=json.dumps(data, ensure_ascii=False)
                            )
                            session.add(record)
                            session.commit()
                            st.success(f"✅ Dane zapisane do bazy (ID: {record.id})")
                        except SQLAlchemyError as e:
                            st.error(f"❌ Błąd zapisu do bazy: {str(e)}")
                            session.rollback()
                        
                        st.download_button(
                            "💾 Pobierz JSON",
                            json.dumps(data, ensure_ascii=False, indent=2),
                            file_name=f"{Path(up.name).stem}_{file_hash}.json",
                            mime="application/json",
                        )
                    else:
                        st.warning("Nie udało się wyekstraktować żadnych danych.")
                        
            except DataExtractionError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                st.error(f"❌ Nieoczekiwany błąd podczas ekstrakcji AI: {str(e)}")
                
    except ValidationError as e:
        st.error(f"❌ Błąd walidacji: {str(e)}")
    except PDFProcessingError as e:
        st.error(f"❌ {str(e)}")
    except DataExtractionError as e:
        st.error(f"❌ {str(e)}")
    except Exception as e:
        st.error(f"❌ Nieoczekiwany błąd inicjalizacji AI: {str(e)}")
