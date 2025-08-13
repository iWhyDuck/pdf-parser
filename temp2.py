# app.py – Streamlit demo z dwoma trybami wyciągania danych z PDF-ów:
# 1) Classic (Regex)   2) AI (GPT-3.5 quick-scan → detailed)
# zależności: streamlit, pdfplumber, openai, sqlalchemy

from __future__ import annotations
import io, json, os, re
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Tuple

import pdfplumber
import streamlit as st
from openai import OpenAI

# Dodane importy dla bazy danych
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ──────────────────────── 0. Konfiguracja ────────────────────────
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
# dodaj własne pola, regexy lub zamień na yaml / json konfig.

# Dodana konfiguracja bazy danych
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
    engine = create_engine("sqlite:///extractions.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()

# ──────────────────────── 1. Extractor classic ─────────────────────
class ClassicExtractor:
    def __init__(self, config: Dict[str, Dict]):
        self.cfg = {
            k: [re.compile(p, re.I) for p in v["patterns"]] for k, v in config.items()
        }

    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    def run(self, text: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for key, pats in self.cfg.items():
            for pat in pats:
                if m := pat.search(text):
                    out[key] = m.group(1).strip()
                    break
        return out


# ──────────────────────── 2. Extractor AI ─────────────────────────
class AIExtractor:
    def __init__(self, api_key: str):
        self.cli = OpenAI(api_key=api_key)

    # – internal helper
    def _chat(self, messages: List[Dict], **kw) -> str:
        r = self.cli.chat.completions.create(
            model=OPENAI_MODEL, messages=messages, temperature=0, **kw
        )
        return r.choices[0].message.content.strip()

    # step 1 – szybkie wykrycie etykiet
    def discover(self, text: str, max_labels: int = 15) -> List[str]:
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
        return [l.strip() for l in raw.split(",") if 2 < len(l.strip()) < 40][:max_labels]

    # step 2 – dokładna ekstrakcja wybranych pól
    def extract(self, text: str, fields: List[str]) -> Dict[str, str]:
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
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}


# ──────────────────────── 3. UI Streamlit ─────────────────────────
st.set_page_config(page_title="PDF Extractor", layout="wide")
st.title("📄 PDF Extractor — Classic vs AI")

mode = st.radio("Tryb parsera:", ["Classic (Regex)", "AI (GPT-3.5)"], horizontal=True)

if mode.startswith("AI") and not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Ustaw zmienną OPENAI_API_KEY lub dodaj ją w .secrets.toml")
    st.stop()

up = st.file_uploader("Wgraj PDF", type="pdf")
if not up:
    st.stop()

pdf_bytes = up.read()
text = ClassicExtractor.extract_text(pdf_bytes)  # staticmethod call
file_hash = sha256(pdf_bytes).hexdigest()[:6]

# Dodana inicjalizacja sesji bazy danych
session = get_db_session()

# ───────────────────── Classic flow ───────────────────────────────
if mode.startswith("Classic"):
    st.header("🔤 Wynik klasyczny (regex)")
    classic = ClassicExtractor(REGEX_FIELDS)
    data = classic.run(text)
    if data:
        for k, v in data.items():
            st.write(f"**{REGEX_FIELDS[k]['display']}**: {v}")
        
        # Dodany zapis do bazy danych
        record = Extraction(
            filename=up.name,
            file_hash=file_hash,
            extraction_method="classic",
            extracted_data=json.dumps(data, ensure_ascii=False)
        )
        session.add(record)
        session.commit()
        st.success(f"✅ Dane zapisane do bazy (ID: {record.id})")
        
        # Dodane pobieranie JSON
        st.download_button(
            "💾 Pobierz JSON",
            json.dumps(data, ensure_ascii=False, indent=2),
            file_name=f"{Path(up.name).stem}_{file_hash}.json",
            mime="application/json",
        )
    else:
        st.warning("Żadne z konfigurowanych pól nie zostało znalezione.")


# ───────────────────── AI flow ────────────────────────────────────
else:
    api = AIExtractor(os.getenv("OPENAI_API_KEY"))
    st.header("🤖 AI Quick-scan")

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
        result = api.extract(text, selected)
        if not result:
            st.error("Brak danych—model nie zwrócił JSON-a.")
            st.stop()

        st.success("Ekstrakcja zakończona")
        st.json(result)
        
        # Dodany zapis do bazy danych dla AI
        record = Extraction(
            filename=up.name,
            file_hash=file_hash,
            extraction_method="ai",
            extracted_data=json.dumps(result, ensure_ascii=False)
        )
        session.add(record)
        session.commit()
        st.success(f"✅ Dane zapisane do bazy (ID: {record.id})")
        
        st.download_button(
            "💾 Pobierz JSON",
            json.dumps(result, ensure_ascii=False, indent=2),
            file_name=f"{Path(up.name).stem}_{file_hash}.json",
            mime="application/json",
        )
