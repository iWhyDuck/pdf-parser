"""Main application entry point for the PDF Parser Streamlit app.

This module contains the main PDFExtractorApp class and application startup logic.
Run this file with `streamlit run src/app.py` to start the application.
"""

import json
import os
from hashlib import sha256
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from langfuse import observe
except ImportError:
    def observe(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func
        return decorator if args else decorator

from pdf_parser import (
    Config,
    DatabaseManager,
    ExtractionRepository,
    PDFProcessor,
    BatchProcessor,
    UIRenderer,
    DataExtractor,
    ClassicExtractor,
    AIExtractor,
    TextExtractor,
    PDFValidator
)


class PDFExtractorApp:
    """Main Streamlit application class.

    This class orchestrates the entire application flow, coordinates
    all components, and handles the main user interaction logic.
    It serves as the application entry point and controller.

    Attributes:
        db_manager: DatabaseManager for database operations
        repository: ExtractionRepository for data persistence
        processor: PDFProcessor for file processing operations
        batch_processor: BatchProcessor for batch operations
        ui: UIRenderer for user interface operations
    """

    def __init__(self) -> None:
        """Initialize application with all required components.

        Sets up dependency injection for all major components
        and establishes the application architecture.
        """
        self.db_manager: DatabaseManager = DatabaseManager()
        self.repository: ExtractionRepository = ExtractionRepository(self.db_manager)
        self.processor: PDFProcessor = PDFProcessor(self.repository)
        self.batch_processor: BatchProcessor = BatchProcessor(self.processor)
        self.ui: UIRenderer = UIRenderer()

    def run(self) -> None:
        """Run the main application flow.

        Orchestrates the complete user interface flow including
        header rendering, mode selection, and processing workflows.
        Handles early exit conditions for missing configuration.
        """
        self.ui.render_header()

        mode: str = self.ui.render_mode_selector()

        if mode.startswith("AI") and not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ Set OPENAI_API_KEY environment variable or add it to .secrets.toml")
            return

        # Batch processing section
        self._handle_batch_processing(mode)

        # Single file processing section
        self._handle_single_file_processing(mode)

    def _handle_batch_processing(self, mode: str) -> None:
        """Handle batch processing workflow.

        Manages the complete batch processing user interface
        and workflow including file upload, field selection,
        and batch execution.

        Args:
            mode: Selected processing mode string
        """
        uploaded_files, selected_fields = self.ui.render_batch_section()

        if uploaded_files and selected_fields:
            if st.button("Extract All (async batch & save to DB)"):
                try:
                    extractor: DataExtractor = self._create_extractor(mode)
                    method: str = "classic" if mode.startswith("Classic") else "ai"

                    batch_results: List[Dict[str, Any]] = self.batch_processor.process_batch(
                        uploaded_files, extractor, selected_fields, method
                    )

                    self._display_batch_results(batch_results)

                except Exception as e:
                    st.error(f"❌ Batch processing error: {str(e)}")

    def _handle_single_file_processing(self, mode: str) -> None:
        """Handle single file processing workflow.

        Manages the complete single file processing user interface
        and workflow, delegating to mode-specific handlers.

        Args:
            mode: Selected processing mode string
        """
        uploaded_file: Any = self.ui.render_single_file_uploader()
        if not uploaded_file:
            return

        try:
            if mode.startswith("Classic"):
                self._handle_classic_mode(uploaded_file)
            else:
                self._handle_ai_mode(uploaded_file)

        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")

    def _handle_classic_mode(self, uploaded_file: Any) -> None:
        """Handle classic regex extraction mode workflow.

        Manages the user interface and processing flow for
        regex-based extraction including field selection.

        Args:
            uploaded_file: Uploaded file object from Streamlit
        """
        st.header("🔤 Available fields (regex)")

        selected_fields: List[str] = self.ui.field_selector.render_field_checkboxes(
            Config.REGEX_FIELDS, "classic"
        )

        if st.button("Extract selected fields", key="single_extract_button"):
            self._process_single_file(uploaded_file, "Classic", selected_fields)

    def _handle_ai_mode(self, uploaded_file: Any) -> None:
        """Handle AI extraction mode workflow.

        Manages the user interface and processing flow for
        AI-based extraction including dynamic field discovery.

        Args:
            uploaded_file: Uploaded file object from Streamlit
        """
        pdf_bytes: bytes = uploaded_file.read()
        PDFValidator.validate_pdf_file(pdf_bytes, uploaded_file.name)
        text: str = TextExtractor.extract_text(pdf_bytes)

        extractor: AIExtractor = AIExtractor(os.getenv("OPENAI_API_KEY"))
        st.header("🤖 AI Quick-scan")

        with st.spinner("Scanning document..."):
            labels: List[str] = extractor.discover_labels(text)

        if not labels:
            st.warning("Model did not identify any labels.")
            return

        selected_fields: List[str] = self.ui.field_selector.render_ai_field_checkboxes(
            labels, "ai_field"
        )

        if st.button("Extract selected fields", key="single_ai_extract_button"):
            try:
                file_hash: str = sha256(pdf_bytes).hexdigest()[:6]
                data: Dict[str, str] = extractor.extract(text, selected_fields)

                if data:
                    db_id: int = self.processor.save_extraction_result(
                        uploaded_file.name, file_hash, "ai", data
                    )
                    self.ui.render_extraction_results(
                        data, uploaded_file.name, file_hash, db_id
                    )
                else:
                    st.warning("Failed to extract any data.")

            except Exception as e:
                st.error(f"❌ AI extraction error: {str(e)}")

    def _process_single_file(self, uploaded_file: Any, mode: str, selected_fields: List[str]) -> None:
        """Process a single uploaded file with specified parameters.

        Executes the complete processing workflow for a single file
        including validation, extraction, and result display.

        Args:
            uploaded_file: Uploaded file object from Streamlit
            mode: Processing mode string
            selected_fields: List of fields to extract
        """
        if not selected_fields:
            st.warning("Select at least one field for extraction.")
            return

        try:
            with st.spinner("Extracting data..."):
                pdf_bytes: bytes = uploaded_file.read()
                extractor: DataExtractor = self._create_extractor(mode)

                data, file_hash = self.processor.process_file(
                    pdf_bytes, uploaded_file.name, extractor, selected_fields
                )

                if data:
                    method: str = "classic" if mode.startswith("Classic") else "ai"
                    db_id: int = self.processor.save_extraction_result(
                        uploaded_file.name, file_hash, method, data
                    )
                    self.ui.render_extraction_results(
                        data, uploaded_file.name, file_hash, db_id
                    )
                else:
                    st.warning("None of the selected fields were found.")

        except Exception as e:
            st.error(f"❌ Extraction error: {str(e)}")

    def _create_extractor(self, mode: str) -> DataExtractor:
        """Factory method for creating appropriate extractor instances.

        Creates and returns the appropriate extractor implementation
        based on the selected processing mode.

        Args:
            mode: Processing mode string

        Returns:
            Appropriate DataExtractor implementation instance
        """
        if mode.startswith("Classic"):
            return ClassicExtractor()
        else:
            return AIExtractor(os.getenv("OPENAI_API_KEY"))

    def _display_batch_results(self, batch_results: List[Dict[str, Any]]) -> None:
        """Display batch processing results interface.

        Shows summary of batch processing results including
        successful extractions, errors, and download options.

        Args:
            batch_results: List of processing results for each file
        """
        st.success("✅ All files processed and saved!")
        st.subheader("Batch extraction results")

        for batch_result in batch_results:
            st.write(f"**File:** {batch_result['file']}")
            if "result" in batch_result:
                st.json(batch_result["result"])
                st.write(f"Saved to database (ID: {batch_result['db_id']})")
            else:
                st.error(f"Error: {batch_result['error']}")

        st.download_button(
            "💾 Download batch JSON",
            json.dumps(batch_results, ensure_ascii=False, indent=2),
            file_name="batch_results.json",
            mime="application/json",
        )


def main() -> None:
    """Main entry point for the application."""
    app = PDFExtractorApp()
    app.run()


if __name__ == "__main__":
    main()
