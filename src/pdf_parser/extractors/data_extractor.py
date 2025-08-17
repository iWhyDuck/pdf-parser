"""Data extractor base class for the PDF parser application.

This module contains the abstract DataExtractor base class that defines
the interface for all data extraction implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from ..exceptions import DataExtractionError

__all__ = ["DataExtractor"]


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
