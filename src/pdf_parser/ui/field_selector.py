"""Field selector component for the PDF parser application.

This module contains the FieldSelector class for rendering field selection
interfaces in the Streamlit application.
"""

from typing import Dict, List, Union

import streamlit as st

__all__ = ["FieldSelector"]


class FieldSelector:
    """UI component for field selection interfaces.

    This class provides reusable methods for rendering field selection
    checkboxes in the Streamlit interface. Supports both predefined
    regex fields and dynamically discovered AI fields.
    """

    @staticmethod
    def render_field_checkboxes(fields_config: Dict[str, Dict[str, Union[str, List[str]]]],
                               key_prefix: str) -> List[str]:
        """Render checkboxes for predefined fields and return selected ones.

        Creates a grid layout of checkboxes for field selection.
        All fields are selected by default for user convenience.

        Args:
            fields_config: Dictionary containing field definitions with display names
            key_prefix: Unique prefix for checkbox keys to avoid conflicts

        Returns:
            List of selected field names
        """
        cols = st.columns(3)
        selected_fields: List[str] = []
        field_keys: List[str] = list(fields_config.keys())

        for i, field_key in enumerate(field_keys):
            display_name: str = fields_config[field_key]['display']
            if cols[i % 3].checkbox(display_name, True, key=f"{key_prefix}_{field_key}"):
                selected_fields.append(field_key)

        return selected_fields

    @staticmethod
    def render_ai_field_checkboxes(labels: List[str], key_prefix: str) -> List[str]:
        """Render checkboxes for AI-discovered fields and return selected ones.

        Creates a grid layout of checkboxes for dynamically discovered fields.
        All fields are selected by default for user convenience.

        Args:
            labels: List of field labels discovered by AI
            key_prefix: Unique prefix for checkbox keys to avoid conflicts

        Returns:
            List of selected field labels
        """
        cols = st.columns(3)
        selected_fields: List[str] = []

        for i, label in enumerate(labels):
            if cols[i % 3].checkbox(label, True, key=f"{key_prefix}_{label}"):
                selected_fields.append(label)

        return selected_fields
