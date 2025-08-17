"""Configuration module for the PDF parser application.

This module contains all configuration parameters including
model settings, file size limits, database configuration,
and regex field definitions.
"""

from typing import Dict, List, Union

__all__ = ["Config"]


class Config:
    """Configuration class containing application settings and constants.

    This class centralizes all configuration parameters including
    model settings, file size limits, database configuration,
    and regex field definitions.
    """

    # OpenAI model configuration
    OPENAI_MODEL: str = "gpt-3.5-turbo-1106"

    # File validation limits
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB maximum file size
    MIN_FILE_SIZE: int = 100  # 100 bytes minimum file size

    # Database configuration
    DATABASE_URL: str = "sqlite:///extractions.db"

    # Regex field definitions for classic extraction
    REGEX_FIELDS: Dict[str, Dict[str, Union[str, List[str]]]] = {
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
