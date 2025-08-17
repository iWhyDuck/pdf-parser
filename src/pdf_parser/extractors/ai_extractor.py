"""AI extractor for the PDF parser application.

This module contains the AIExtractor class for AI-powered data
extraction using OpenAI GPT models.
"""

import json
import re
from typing import Any, Dict, List

from openai import OpenAI, OpenAIError

try:
    from langfuse import observe
except ImportError:
    def observe(*args: Any, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func
        return decorator if args else decorator

from ..config import Config
from ..exceptions import DataExtractionError
from .data_extractor import DataExtractor

__all__ = ["AIExtractor"]


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
