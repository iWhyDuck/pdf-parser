"""Text extractor for the PDF parser application.

This module contains the TextExtractor class for extracting text content
from PDF files using the pdfplumber library.
"""

import io
from typing import List, Optional

import pdfplumber
import streamlit as st

from ..exceptions import PDFProcessingError

__all__ = ["TextExtractor"]


class TextExtractor:
    """Extracts text content from PDF files.

    This class provides static methods for extracting text from PDF files
    using the pdfplumber library. Handles multi-page documents and provides
    error recovery for individual page processing failures.
    """

    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        """Extract text content from PDF file.

        Processes all pages in the PDF document and extracts text content.
        Continues processing even if individual pages fail, providing
        partial results when possible.

        Args:
            pdf_bytes: Raw PDF file content as bytes

        Returns:
            Extracted text content as a single string with page breaks

        Raises:
            PDFProcessingError: If PDF cannot be opened or contains no text
        """
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                if not pdf.pages:
                    raise PDFProcessingError("PDF contains no pages")

                text_parts: List[str] = []
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text: Optional[str] = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        st.warning(f"⚠️ Failed to process page {i+1}: {str(e)}")
                        continue

                if not text_parts:
                    raise PDFProcessingError("Failed to extract text from any page")

                return "\n".join(text_parts)

        except Exception as e:
            if isinstance(e, PDFProcessingError):
                raise
            raise PDFProcessingError(f"PDF reading error: {str(e)}")
