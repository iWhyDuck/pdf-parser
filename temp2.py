# app.py – minimalistyczny PDF-parser z trybem „Classic (Regex)”
# oraz dwustopniowym trybem „AI (quick-scan → detailed)”
# wymaga: pdfplumber, streamlit ≥1.26, openai ≥1.14

from __future__ import annotations

import io
import json
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Tuple

import pdfplumber
import streamlit as st
from openai import OpenAI

# ─────────────────────────────── 0. ENV ───────────────────────────────────
os.environ.setdefault("OPENAI_DISABLE_HTTP2", "true")          # proxy-safety
OPENAI_MODEL = "gpt-3.5-turbo-1106"

# ─────────────────────────────── 1. KONFIG ────────────────────────────────
EXTRACTION_FIELDS: Dict[str, Dict] = {
    "customer_name": {
        "display": "Customer Name",
        "patterns": [r"Customer Name[:\s]*([A-Za-z ,.'-]+)"],
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
# pola i regexy możesz swobodnie rozszerzyć

# ─────────────────────────────── 2. KLASY ────────────────────────────────
class ClassicExtractor:
    """Prosty ekstraktor regex-owy."""
    def __init__(self, config: Dict[str, Dict]):
        self.cfg = {
            k: [re.compile(p, re.I) for p in v["patterns"]] for k, v in config.items()
        }

    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)

    def find(self, text: str) -> Dict[str, str]:
        found: Dict[str, str] = {}
        for key, patterns in self.cfg.items():
            for pat in patterns:
                if m := pat.search(text):
                    found[key] = m.group(1).strip()
                    break
        return found


class AIExtractor:
    """Dwustopniowa ekstrakcja: quick-scan → detailed."""
    def __init__(self, api_key: str):
        self.cli = OpenAI(api_key=api_key)

    # ––––– helpers –––––
    def _chat(self, messages: List[Dict], **params) -> str:
        resp = self.cli.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0,
            **params,
        )
        return resp.choices[0].message.content.strip()

    # step 1 – nazwy pól
    def discover_fields(self, text: str) -> List[str]:
        prompt = (
            "List comma-separated labels (no values) that look like data fields "
            "in the document below. 15 names max.\n\n" + text[:3000]
        )
        raw = self._chat(
            [
                {"role": "system", "content": "Return only the labels."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
        )
        return [f.strip() for f in raw.split(",") if 2 < len(f.strip()) < 40][:15]

    # step 2 – wartości pól
    def extract(self, text: str, fields: List[str]) -> Dict[str, str]:
        prompt = (
            f"Extract the following fields: {', '.join(fields)}\n\n"
            "Return ONLY minified JSON {\"Field\":\"Value\"}. "
            "If a field is missing use null.\n\n"
            + text[:20_000]
        )
        raw = self._chat(
            [
                {"role": "system", "content": "You are a data-extraction engine."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
        )
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}


# ───────────────────────────── 3. STRUMIENIE ───────────────────────────────
st.set_page_config(page_title="PDF Extractor", layout="wide")
st.title("📄 PDF Extractor – Classic vs AI")

mode = st.radio("Choose parser:", ["Classic (Regex)", "AI (GPT-3.5)"], horizontal=True)

if mode.startswith("AI") and not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️  Set OPENAI_API_KEY in environment or .streamlit/secrets.toml")
    st.stop()

file = st.file_uploader("Upload PDF", type="pdf")

if not file:
    st.stop()

pdf_bytes = file.read()
file_hash = sha256(pdf_bytes).hexdigest()
txt = ClassicExtractor.extract_text(pdf_bytes)        # staticmethod call

# ────────────────── Classic branch ──────────────────
if mode.startswith("Classic"):
    extractor = ClassicExtractor(EXTRACTION_FIELDS)
    data = extractor.find(txt)
    if data:
        st.subheader("🔤 Classic extraction")
        for k, v in data.items():
            st.write(f"**{EXTRACTION_FIELDS[k]['display']}**: {v}")
    else:
        st.warning("No configured fields found.")

# ─────────────────── AI branch ───────────────────────
else:
    api = AIExtractor(os.getenv("OPENAI_API_KEY"))

    # 1️⃣ quick-scan
    st.subheader("🤖 AI quick-scan")
    labels = api.discover_fields(txt)
    if not labels:
        st.warning("GPT did not detect any fields.")
        st.stop()

    cols = st.columns(3)
    selected: List[str] = []
    for i, label in enumerate(labels):
        if cols[i % 3].checkbox(label, True):
            selected.append(label)

    if st.button("Extract selected fields"):
        # 2️⃣ detailed extraction
        result = api.extract(txt, selected)
        if not result:
            st.error("AI returned no JSON.")
            st.stop()

        st.success("Extraction complete")
        st.json(result)
        st.download_button(
            "💾 Download JSON",
            json.dumps(result, ensure_ascii=False, indent=2),
            file_name=f"{Path(file.name).stem}_{file_hash[:6]}.json",
            mime="application/json",
        )
