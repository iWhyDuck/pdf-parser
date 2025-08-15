# app.py – Streamlit demo z dwoma trybami wyciągania danych z PDF-ów:
# 1) Classic (Regex)   2) AI (GPT-3.5 quick-scan → detailed)
# zależności: streamlit, pdfplumber, openai, sqlalchemy

from __future__ import annotations
import io, json, os, re
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pdfplumber
import streamlit as st
from openai import OpenAI
from openai import OpenAIError

# Dodane importy dla bazy danych
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, func
from sqlalchemy.orm import declarative_base
Base = declarative_base()
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

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

mode = st.radio("Tryb parsera:", ["Classic (Regex)", "AI (GPT-3.5)"], horizontal=True)

if mode.startswith("AI") and not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Ustaw zmienną OPENAI_API_KEY lub dodaj ją w .secrets.toml")
    st.stop()

up = st.file_uploader("Wgraj PDF", type="pdf")
if not up:
    st.stop()

# ──────────────────────── 7. Główny proces z error handling ─────────────────────
try:
    # Wczytaj i zwaliduj plik
    pdf_bytes = up.read()
    PDFValidator.validate_pdf_file(pdf_bytes, up.name)
    
    # Ekstraktuj tekst
    with st.spinner("Przetwarzanie PDF..."):
        text = ClassicExtractor.extract_text(pdf_bytes)
    
    file_hash = sha256(pdf_bytes).hexdigest()[:6]
    
    # Inicjalizacja sesji bazy danych
    try:
        session = get_db_session()
    except DatabaseError as e:
        st.error(f"❌ {str(e)}")
        st.stop()

except ValidationError as e:
    st.error(f"❌ Błąd walidacji: {str(e)}")
    st.stop()
except PDFProcessingError as e:
    st.error(f"❌ {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"❌ Nieoczekiwany błąd: {str(e)}")
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

    if st.button("Extract selected fields"):
        try:
            with st.spinner("Ekstraktowanie danych..."):
                classic = ClassicExtractor(REGEX_FIELDS)
                data = classic.run(text, selected)
                
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
            if cols[i % 3].checkbox(lab, True):
                selected.append(lab)

        if st.button("Extract selected fields"):
            try:
                with st.spinner("Ekstraktowanie danych przez AI..."):
                    result = api.extract(text, selected)
                    
                    if result:
                        st.success("Ekstrakcja zakończona")
                        st.json(result)
                        
                        # Zapis do bazy danych z obsługą błędów
                        try:
                            record = Extraction(
                                filename=up.name,
                                file_hash=file_hash,
                                extraction_method="ai",
                                extracted_data=json.dumps(result, ensure_ascii=False)
                            )
                            session.add(record)
                            session.commit()
                            st.success(f"✅ Dane zapisane do bazy (ID: {record.id})")
                        except SQLAlchemyError as e:
                            st.error(f"❌ Błąd zapisu do bazy: {str(e)}")
                            session.rollback()
                        
                        st.download_button(
                            "💾 Pobierz JSON",
                            json.dumps(result, ensure_ascii=False, indent=2),
                            file_name=f"{Path(up.name).stem}_{file_hash}.json",
                            mime="application/json",
                        )
                    else:
                        st.warning("Nie udało się wyekstraktować żadnych danych.")
                        
            except DataExtractionError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                st.error(f"❌ Nieoczekiwany błąd podczas ekstrakcji AI: {str(e)}")
                
    except DataExtractionError as e:
        st.error(f"❌ {str(e)}")
    except Exception as e:
        st.error(f"❌ Nieoczekiwany błąd inicjalizacji AI: {str(e)}")
