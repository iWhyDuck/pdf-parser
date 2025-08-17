"""Extractors module for the PDF parser application.

This module contains text and data extraction classes including
text extraction from PDFs, regex-based classic extraction,
and AI-powered extraction using OpenAI models.
"""

import io
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pdfplumber
import streamlit as st
from openai import OpenAI, OpenAIError

try:
    from langfuse import observe
except ImportError:
    def observe(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func
        return decorator if args else decorator

from ..config import Config
from ..exceptions import DataExtractionError, PDFProcessingError

__all__ = ["TextExtractor", "DataExtractor", "ClassicExtractor", "AIExtractor"]


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


class DataExtractor(ABC):
    """Abstract base class for data extraction implementations.

    This abstract class defines the interface that all data extractors
    must implement. It follows the Strategy pattern to allow different
    extraction methods to be used interchangeably.
    """

    @abstractmethod
    def extract(self, text: str, fields: List[str]) -> Dict[str, str]:
        """Extract data fields from text content.

        Abstract method that must be implemented by concrete extractors
        to perform field extraction from the provided text.

        Args:
            text: Input text content to extract data from
            fields: List of field names to extract

        Returns:
            Dictionary mapping field names to extracted values

        Raises:
            DataExtractionError: If extraction process fails
        """
        pass


class ClassicExtractor(DataExtractor):
    """Regex-based data extractor for structured documents.

    This extractor uses predefined regular expression patterns to locate
    and extract specific data fields from text content. It's optimized
    for documents with consistent formatting and known field patterns.

    Attributes:
        cfg: Compiled regex patterns organized by field name
    """

    def __init__(self, config: Dict[str, Dict[str, Any]] = None) -> None:
        """Initialize extractor with regex configuration.

        Compiles all regex patterns for efficient repeated use.
        Patterns are compiled with case-insensitive flag.

        Args:
            config: Dictionary containing field definitions and regex patterns

        Raises:
            DataExtractionError: If regex compilation fails
        """
        if config is None:
            config = Config.REGEX_FIELDS

        try:
            self.cfg: Dict[str, List[Any]] = {
                k: [re.compile(p, re.I) for p in v["patterns"]]
                for k, v in config.items()
            }
        except re.error as e:
            raise DataExtractionError(f"Regex compilation error: {str(e)}")

    @observe(name="classic_extraction")
    def extract(self, text: str, fields: Optional[List[str]] = None) -> Dict[str, str]:
        """Extract data using regex patterns.

        Applies regex patterns to extract specified fields from text.
        If no fields are specified, attempts to extract all configured fields.

        Args:
            text: Input text content to extract data from
            fields: Optional list of specific fields to extract

        Returns:
            Dictionary mapping field names to extracted values

        Raises:
            DataExtractionError: If text is empty or extraction fails
        """
        if not text or not text.strip():
            raise DataExtractionError("No text content to process")

        try:
            out: Dict[str, str] = {}
            fields_to_extract: List[str] = fields if fields else list(self.cfg.keys())

            for key in fields_to_extract:
                if key not in self.cfg:
                    st.warning(f"⚠️ Unknown field: {key}")
                    continue

                value: Optional[str] = self._extract_field_value(key, text)
                if value:
                    out[key] = value

            return out

        except Exception as e:
            raise DataExtractionError(f"Classic extraction error: {str(e)}")

    def _extract_field_value(self, key: str, text: str) -> Optional[str]:
        """Extract value for a specific field using regex patterns.

        Tries each regex pattern for the field until a match is found.
        Returns the first successful match or None if no patterns match.

        Args:
            key: Field name to extract
            text: Text content to search in

        Returns:
            Extracted field value or None if not found
        """
        for pat in self.cfg[key]:
            try:
                match = pat.search(text)
                if match:
                    return match.group(1).strip()
            except Exception as e:
                st.warning(f"⚠️ Error processing field '{key}': {str(e)}")
                continue
        return None


class AIExtractor(DataExtractor):
    """AI-powered data extractor using OpenAI GPT models.

    This extractor uses OpenAI's language models to identify and extract
    data fields from unstructured text. It can discover field labels
    dynamically and extract values using natural language understanding.

    Attributes:
        cli: OpenAI client instance for API communication
    """

    def __init__(self, api_key: str) -> None:
        """Initialize AI extractor with OpenAI API key.

        Creates OpenAI client instance for subsequent API calls.

        Args:
            api_key: OpenAI API key for authentication

        Raises:
            DataExtractionError: If API key is missing or client creation fails
        """
        if not api_key:
            raise DataExtractionError("Missing OpenAI API key")

        try:
            self.cli: OpenAI = OpenAI(api_key=api_key)
        except Exception as e:
            raise DataExtractionError(f"OpenAI client initialization error: {str(e)}")

    @observe(name="openai_chat_completion", as_type="generation")
    def _chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Execute OpenAI chat completion request.

        Sends messages to OpenAI API and returns the response content.
        Configured with zero temperature for consistent outputs.

        Args:
            messages: List of message dictionaries for the conversation
            **kwargs: Additional parameters for the API call

        Returns:
            Generated response content as string

        Raises:
            DataExtractionError: If API call fails or returns empty response
        """
        try:
            response = self.cli.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=messages,
                temperature=0,
                **kwargs
            )
            if not response.choices or not response.choices[0].message.content:
                raise DataExtractionError("OpenAI returned empty response")

            return response.choices[0].message.content.strip()

        except OpenAIError as e:
            raise DataExtractionError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise DataExtractionError(f"Unexpected error during AI call: {str(e)}")

    @observe(name="discover_labels")
    def discover_labels(self, text: str, max_labels: int = 15) -> List[str]:
        """Discover potential field labels in document text.

        Uses AI to identify form field names and labels that appear
        in the document content. Useful for dynamic field discovery
        in unknown document formats.

        Args:
            text: Document text content to analyze
            max_labels: Maximum number of labels to return

        Returns:
            List of discovered field label strings

        Raises:
            DataExtractionError: If text is empty or analysis fails
        """
        if not text or not text.strip():
            raise DataExtractionError("No text content to analyze")

        try:
            prompt: str = (
                "Return comma-separated labels (no values) that look like form-field names "
                f"in the document below (≤{max_labels}).\n\n{text[:3000]}"
            )
            raw_response: str = self._chat(
                [
                    {"role": "system", "content": "You are PDF-data assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
            )

            if not raw_response:
                return []

            labels: List[str] = [
                label.strip() for label in raw_response.split(",")
                if 2 < len(label.strip()) < 40
            ]
            return labels[:max_labels]

        except Exception as e:
            if isinstance(e, DataExtractionError):
                raise
            raise DataExtractionError(f"Label discovery error: {str(e)}")

    @observe(name="ai_extract_data")
    def extract(self, text: str, fields: List[str]) -> Dict[str, str]:
        """Extract specified fields using AI analysis.

        Uses OpenAI to analyze text content and extract values for
        the specified fields. Returns structured data as JSON.

        Args:
            text: Input text content to extract data from
            fields: List of field names to extract

        Returns:
            Dictionary mapping field names to extracted values

        Raises:
            DataExtractionError: If text is empty, no fields specified, or extraction fails
        """
        if not text or not text.strip():
            raise DataExtractionError("No text content to process")

        if not fields:
            raise DataExtractionError("No fields specified for extraction")

        try:
            prompt: str = self._build_extraction_prompt(fields, text)
            raw_response: str = self._chat(
                [
                    {"role": "system", "content": "You are data-extraction engine."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
            )

            return self._parse_extraction_result(raw_response)

        except Exception as e:
            if isinstance(e, DataExtractionError):
                raise
            raise DataExtractionError(f"AI extraction error: {str(e)}")

    def _build_extraction_prompt(self, fields: List[str], text: str) -> str:
        """Build prompt for AI extraction request.

        Constructs a detailed prompt that instructs the AI model
        to extract specific fields and return results in JSON format.

        Args:
            fields: List of field names to extract
            text: Document text content (truncated to 20k characters)

        Returns:
            Formatted prompt string for AI model
        """
        return (
            f"Extract: {', '.join(fields)}\n\n"
            "Return ONLY compact JSON {\"Field\":\"Value\"}. "
            "If a field is missing, set null.\n\n"
            + text[:20_000]
        )

    def _parse_extraction_result(self, raw_response: str) -> Dict[str, str]:
        """Parse AI response to extract JSON data.

        Locates JSON content in the AI response and parses it
        into a dictionary structure for field-value mapping.

        Args:
            raw_response: Raw response text from AI model

        Returns:
            Parsed dictionary containing extracted field-value pairs

        Raises:
            DataExtractionError: If JSON parsing fails or format is invalid
        """
        match = re.search(r"\{.*\}", raw_response, re.S)
        if not match:
            raise DataExtractionError("AI did not return valid JSON")

        try:
            result: Any = json.loads(match.group(0))
            if not isinstance(result, dict):
                raise DataExtractionError("AI returned invalid data format")

            return result

        except json.JSONDecodeError as e:
            raise DataExtractionError(f"JSON parsing error from AI response: {str(e)}")
