# app.py – on-demand PDF field extractor (SQLite + file hash)
# autor: (Twoje imię)

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pdfplumber
import streamlit as st
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text, create_engine)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ─────────────────────────────── 1. KONFIG ────────────────────────────────
EXTRACTION_FIELDS: Dict[str, Dict] = {
    "customer_name": {
        "display_name": "Customer Name",
        "patterns": [r"Customer Name[:\s]*([A-Za-z ,.'-]+)", r"Name[:\s]*([A-Za-z ,.'-]+)"],
        "default_selected": True,
    },
    "customer_email": {
        "display_name": "E-mail",
        "patterns": [r"Email[:\s]*([\w\.-]+@[\w\.-]+\.\w+)", r"E-mail[:\s]*([\w\.-]+@[\w\.-]+\.\w+)"],
        "default_selected": True,
    },
    "branch_name": {
        "display_name": "Branch / Office",
        "patterns": [r"Branch[:\s]*([A-Za-z0-9 &'-]+)", r"Office[:\s]*([A-Za-z0-9 &'-]+)"],
        "default_selected": True,
    },
    "claim_type": {
        "display_name": "Claim Type",
        "patterns": [r"Claim Type[:\s]*([A-Za-z ]+)", r"Type[:\s]*([A-Za-z ]+)"],
        "default_selected": True,
    },
    "claim_amount": {
        "display_name": "Claim Amount",
        "patterns": [r"Amount[:\s]*\$?([\d,]+\.\d{2})", r"Total[:\s]*\$?([\d,]+\.\d{2})"],
        "default_selected": False,
    },
}

TEMPLATE_FILE = Path.home() / ".field_templates.json"

# ───────────────────────────── 2. BAZA DANYCH ─────────────────────────────
engine = create_engine("sqlite:///extractions.db", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    uploaded_at = Column(DateTime, default=dt.datetime.utcnow)
    status = Column(String(30), default="parsed")
    fields = relationship("ExtractedField", back_populates="document", cascade="all, delete-orphan")


class ExtractedField(Base):
    __tablename__ = "extracted_fields"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    name = Column(String(100), nullable=False)
    value = Column(Text)
    confidence = Column(Float)
    selected = Column(Boolean, default=True)
    document = relationship("Document", back_populates="fields")


Base.metadata.create_all(engine)

# ───────────────────────────── 3. EKSTRAKTOR ──────────────────────────────
class PDFFieldExtractor:
    def __init__(self, config: Dict[str, Dict]):
        self.config = config
        for spec in self.config.values():
            spec["compiled"] = [re.compile(p, re.IGNORECASE) for p in spec["patterns"]]

    @staticmethod
    def _match_first(text: str, patterns: List[re.Pattern]) -> Tuple[str | None, int]:
        for idx, pattern in enumerate(patterns):
            if (m := pattern.search(text)):
                return m.group(1).strip(), idx
        return None, -1

    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)

    def find_fields(self, text: str) -> Dict[str, Dict]:
        """Zwraca tylko te pola, które rzeczywiście występują w dokumencie."""
        found: Dict[str, Dict] = {}
        for key, spec in self.config.items():
            value, idx = self._match_first(text, spec["compiled"])
            if value:
                found[key] = {
                    "display_name": spec["display_name"],
                    "value": value,
                    "confidence": 1.0 - 0.1 * idx,
                    "default_selected": spec.get("default_selected", True),
                }
        return found


extractor = PDFFieldExtractor(EXTRACTION_FIELDS)

# ───────────────────────────── 4. STREAMLIT UI ────────────────────────────
st.set_page_config(page_title="PDF Data Extractor", layout="wide")
st.title("PDF Data Extractor – on-demand workflow")

uploaded = st.file_uploader("Upload PDF", type="pdf")

if uploaded:
    pdf_bytes = uploaded.read()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Analiza PDF (tylko raz per plik)
    if st.session_state.get("hash") != file_hash:
        st.session_state["text"] = extractor.extract_text(pdf_bytes)
        st.session_state["found"] = extractor.find_fields(st.session_state["text"])
        st.session_state["hash"] = file_hash
        st.session_state["extracted_now"] = False  # reset po zmianie pliku

    found_fields = st.session_state["found"]

    if not found_fields:
        st.warning("No configured fields found in this document.")
        st.stop()

    # FORM: wybór pól
    with st.form("selector"):
        st.subheader("Select fields to extract")
        selected_keys = []
        for key, meta in found_fields.items():
            default = meta["default_selected"]
            label = f'{meta["display_name"]} ({meta["confidence"]:.0%})'
            if st.checkbox(label, value=default, key=f"chk_{key}"):
                selected_keys.append(key)
        submitted = st.form_submit_button("Extract")

    # Po kliknięciu EXTRACT
    if submitted:
        st.session_state["extracted_now"] = True
        report = {k: found_fields[k]["value"] for k in selected_keys}
        df = pd.DataFrame(
            [{"Field": found_fields[k]["display_name"],
              "Value": found_fields[k]["value"],
              "Confidence": f'{found_fields[k]["confidence"]:.0%}'} for k in selected_keys]
        )

        # Zapis do SQLite (nadpisuje wpis o tym samym hash-u)
        with SessionLocal() as db:
            doc = db.query(Document).filter_by(file_hash=file_hash).one_or_none()
            if doc is None:
                doc = Document(filename=uploaded.name, file_hash=file_hash)
                db.add(doc)
                db.flush()
            db.query(ExtractedField).filter_by(document_id=doc.id).delete()
            for k in selected_keys:
                meta = found_fields[k]
                db.add(ExtractedField(
                    document=doc,
                    name=meta["display_name"],
                    value=meta["value"],
                    confidence=meta["confidence"],
                    selected=True
                ))
            db.commit()

    # Wyświetlenie raportu i opcji eksportu po ekstrakcji
    if st.session_state.get("extracted_now"):
        st.success("Extraction complete – results saved to SQLite.")
        st.table(df)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download JSON",
                data=json.dumps(report, indent=2),
                file_name="report.json",
                mime="application/json",
            )
        with col2:
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False),
                file_name="report.csv",
                mime="text/csv",
            )

    # Historia ostatnich 5 plików
    with st.sidebar:
        st.header("Last 5 documents")
        with SessionLocal() as db:
            last_docs = db.query(Document).order_by(Document.uploaded_at.desc()).limit(5).all()
            for d in last_docs:
                st.markdown(f"- {d.filename}  \n<small>{d.uploaded_at:%Y-%m-%d %H:%M:%S}</small>",
                            unsafe_allow_html=True)
else:
    st.info("Upload a PDF file to begin.")
