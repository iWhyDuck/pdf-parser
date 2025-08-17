"""UI renderer component for the PDF parser application.

This module contains the UIRenderer class for rendering the main Streamlit
user interface including headers, mode selection, file upload interfaces,
and result display.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

from ..config import Config
from .field_selector import FieldSelector

__all__ = ["UIRenderer"]


class UIRenderer:
    """Renders the main Streamlit user interface.

    This class handles all UI rendering operations including headers,
    mode selection, file upload interfaces, and result display.
    It coordinates the overall user experience flow.

    Attributes:
        field_selector: FieldSelector instance for field selection UI
    """

    def __init__(self) -> None:
        """Initialize UI renderer with field selector component."""
        self.field_selector: FieldSelector = FieldSelector()

    def render_header(self) -> None:
        """Render application header and configuration status.

        Sets up the page configuration, displays the main title,
        and shows Langfuse monitoring status information.
        """
        st.set_page_config(page_title="PDF Extractor", layout="wide")
        st.title("📄 PDF Extractor — Classic vs AI")

        if self._check_langfuse_config():
            st.info("🔍 Langfuse monitoring active")
        else:
            st.info("📊 Monitoring disabled - add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")

    def render_mode_selector(self) -> str:
        """Render processing mode selection radio buttons.

        Provides user interface for selecting between Classic (regex)
        and AI (GPT) extraction modes.

        Returns:
            Selected mode string
        """
        return st.radio("Parser mode:", ["Classic (Regex)", "AI (GPT-3.5)"], horizontal=True)

    def render_batch_section(self) -> Tuple[List[Any], List[str]]:
        """Render batch processing section interface.

        Displays file uploader for multiple files and field selection
        interface for batch processing operations.

        Returns:
            Tuple containing uploaded files list and selected fields list
        """
        st.header("📦 Batch PDF Extraction (async database saving)")

        uploaded_files: List[Any] = st.file_uploader(
            "Upload multiple PDF files",
            type="pdf",
            accept_multiple_files=True,
            key="batch_uploader"
        )

        selected_fields: List[str] = []
        if uploaded_files:
            st.subheader("🔤 Available fields for batch extraction")
            selected_fields = self.field_selector.render_field_checkboxes(
                Config.REGEX_FIELDS, "batch"
            )

        return uploaded_files, selected_fields

    def render_single_file_uploader(self) -> Any:
        """Render single file upload interface.

        Provides file uploader widget for single PDF file processing.

        Returns:
            Uploaded file object or None if no file uploaded
        """
        return st.file_uploader("Upload PDF", type="pdf", key="single_uploader")

    def render_extraction_results(self, data: Dict[str, str], filename: str,
                                file_hash: str, db_id: int) -> None:
        """Render extraction results display interface.

        Shows extracted field-value pairs, database save confirmation,
        and provides download button for JSON results.

        Args:
            data: Extracted field-value pairs
            filename: Original PDF filename
            file_hash: SHA256 hash of file content
            db_id: Database ID of saved extraction record
        """
        st.success("Extraction completed")

        for key, value in data.items():
            display_name: str = Config.REGEX_FIELDS.get(key, {}).get('display', key)
            st.write(f"**{display_name}**: {value}")

        st.success(f"✅ Data saved to database (ID: {db_id})")

        st.download_button(
            "💾 Download JSON",
            json.dumps(data, ensure_ascii=False, indent=2),
            file_name=f"{Path(filename).stem}_{file_hash}.json",
            mime="application/json",
        )

    def _check_langfuse_config(self) -> bool:
        """Check if Langfuse monitoring is properly configured.

        Verifies that required environment variables for Langfuse
        monitoring are present and non-empty.

        Returns:
            True if Langfuse is properly configured, False otherwise
        """
        required_keys: List[str] = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
        missing_keys: List[str] = [key for key in required_keys if not os.getenv(key)]
        return len(missing_keys) == 0
