"""Classic extractor for the PDF parser application.

This module contains the ClassicExtractor class for regex-based data
extraction from structured documents.
"""

import re
from typing import Any, Dict, List, Optional

import streamlit as st

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

__all__ = ["ClassicExtractor"]


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
