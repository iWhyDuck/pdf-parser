"""Tests for the extractors module.

This module contains comprehensive tests for text and data extraction functionality
including text extraction from PDFs, regex-based classic extraction,
and AI-powered extraction using OpenAI models.
"""

import io
import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.pdf_parser.extractors import TextExtractor, ClassicExtractor, AIExtractor
from src.pdf_parser.exceptions import PDFProcessingError, DataExtractionError
from src.pdf_parser.config import Config


class TestTextExtractor:
    """Test cases for TextExtractor class."""

    def test_extract_text_success(self, sample_pdf_bytes):
        """Test successful text extraction from PDF."""
        with patch('src.pdf_parser.extractors.pdfplumber') as mock_pdfplumber:
            # Mock PDF structure
            mock_page = Mock()
            mock_page.extract_text.return_value = "Sample text from PDF"

            mock_pdf = Mock()
            mock_pdf.pages = [mock_page]
            mock_pdf.__enter__ = Mock(return_value=mock_pdf)
            mock_pdf.__exit__ = Mock(return_value=None)

            mock_pdfplumber.open.return_value = mock_pdf

            result = TextExtractor.extract_text(sample_pdf_bytes)

            assert result == "Sample text from PDF"
            mock_pdfplumber.open.assert_called_once()

    def test_extract_text_multiple_pages(self, sample_pdf_bytes):
        """Test text extraction from multi-page PDF."""
        with patch('src.pdf_parser.extractors.pdfplumber') as mock_pdfplumber:
            # Mock multiple pages
            mock_page1 = Mock()
            mock_page1.extract_text.return_value = "Page 1 content"
            mock_page2 = Mock()
            mock_page2.extract_text.return_value = "Page 2 content"

            mock_pdf = Mock()
            mock_pdf.pages = [mock_page1, mock_page2]
            mock_pdf.__enter__ = Mock(return_value=mock_pdf)
            mock_pdf.__exit__ = Mock(return_value=None)

            mock_pdfplumber.open.return_value = mock_pdf

            result = TextExtractor.extract_text(sample_pdf_bytes)

            assert result == "Page 1 content\nPage 2 content"

    def test_extract_text_no_pages(self, sample_pdf_bytes):
        """Test text extraction from PDF with no pages."""
        with patch('src.pdf_parser.extractors.pdfplumber') as mock_pdfplumber:
            mock_pdf = Mock()
            mock_pdf.pages = []
            mock_pdf.__enter__ = Mock(return_value=mock_pdf)
            mock_pdf.__exit__ = Mock(return_value=None)

            mock_pdfplumber.open.return_value = mock_pdf

            with pytest.raises(PDFProcessingError, match="PDF contains no pages"):
                TextExtractor.extract_text(sample_pdf_bytes)

    def test_extract_text_page_extraction_fails(self, sample_pdf_bytes):
        """Test handling of page extraction failures."""
        with patch('src.pdf_parser.extractors.pdfplumber') as mock_pdfplumber, \
             patch('src.pdf_parser.extractors.st') as mock_st:

            # Mock page that fails extraction
            mock_page1 = Mock()
            mock_page1.extract_text.side_effect = Exception("Page extraction failed")

            # Mock successful page
            mock_page2 = Mock()
            mock_page2.extract_text.return_value = "Page 2 content"

            mock_pdf = Mock()
            mock_pdf.pages = [mock_page1, mock_page2]
            mock_pdf.__enter__ = Mock(return_value=mock_pdf)
            mock_pdf.__exit__ = Mock(return_value=None)

            mock_pdfplumber.open.return_value = mock_pdf

            result = TextExtractor.extract_text(sample_pdf_bytes)

            assert result == "Page 2 content"
            mock_st.warning.assert_called_once()

    def test_extract_text_no_text_extracted(self, sample_pdf_bytes):
        """Test handling when no text can be extracted."""
        with patch('src.pdf_parser.extractors.pdfplumber') as mock_pdfplumber:
            mock_page = Mock()
            mock_page.extract_text.return_value = None

            mock_pdf = Mock()
            mock_pdf.pages = [mock_page]
            mock_pdf.__enter__ = Mock(return_value=mock_pdf)
            mock_pdf.__exit__ = Mock(return_value=None)

            mock_pdfplumber.open.return_value = mock_pdf

            with pytest.raises(PDFProcessingError, match="Failed to extract text from any page"):
                TextExtractor.extract_text(sample_pdf_bytes)

    def test_extract_text_pdf_open_fails(self, sample_pdf_bytes):
        """Test handling of PDF opening failures."""
        with patch('src.pdf_parser.extractors.pdfplumber') as mock_pdfplumber:
            mock_pdfplumber.open.side_effect = Exception("Cannot open PDF")

            with pytest.raises(PDFProcessingError, match="PDF reading error"):
                TextExtractor.extract_text(sample_pdf_bytes)


class TestClassicExtractor:
    """Test cases for ClassicExtractor class."""

    def test_init_success(self, test_config):
        """Test successful initialization with custom config."""
        extractor = ClassicExtractor(test_config)
        assert len(extractor.cfg) == 2
        assert "test_field" in extractor.cfg
        assert "number_field" in extractor.cfg

    def test_init_default_config(self):
        """Test initialization with default config."""
        extractor = ClassicExtractor()
        assert len(extractor.cfg) == len(Config.REGEX_FIELDS)
        for field_name in Config.REGEX_FIELDS:
            assert field_name in extractor.cfg

    def test_init_regex_compilation_error(self):
        """Test handling of regex compilation errors."""
        invalid_config = {
            "bad_field": {
                "display": "Bad Field",
                "patterns": ["[invalid regex"]  # Invalid regex pattern
            }
        }
        with pytest.raises(DataExtractionError, match="Regex compilation error"):
            ClassicExtractor(invalid_config)

    def test_extract_success(self, classic_extractor, sample_text_content):
        """Test successful data extraction."""
        fields = ["customer_name", "policy_number"]
        result = classic_extractor.extract(sample_text_content, fields)

        assert "customer_name" in result
        assert "policy_number" in result
        assert "John Smith" in result["customer_name"]
        assert "POL-123456" in result["policy_number"]

    def test_extract_all_fields(self, classic_extractor, sample_text_content):
        """Test extraction of all configured fields."""
        result = classic_extractor.extract(sample_text_content)

        # Should attempt to extract all configured fields
        assert isinstance(result, dict)

    def test_extract_empty_text(self, classic_extractor):
        """Test extraction with empty text."""
        with pytest.raises(DataExtractionError, match="No text content to process"):
            classic_extractor.extract("", ["customer_name"])

    def test_extract_whitespace_only_text(self, classic_extractor):
        """Test extraction with whitespace-only text."""
        with pytest.raises(DataExtractionError, match="No text content to process"):
            classic_extractor.extract("   \n\t   ", ["customer_name"])

    def test_extract_unknown_field(self, classic_extractor, sample_text_content):
        """Test extraction with unknown field."""
        with patch('src.pdf_parser.extractors.st') as mock_st:
            result = classic_extractor.extract(sample_text_content, ["unknown_field"])

            assert result == {}
            mock_st.warning.assert_called_once()

    def test_extract_field_value_success(self, test_config):
        """Test successful field value extraction."""
        extractor = ClassicExtractor(test_config)
        text = "Test Field: Sample Value\nNumber: 12345"

        result = extractor._extract_field_value("test_field", text)
        assert result == "Sample Value"

        result = extractor._extract_field_value("number_field", text)
        assert result == "12345"

    def test_extract_field_value_not_found(self, classic_extractor):
        """Test field value extraction when pattern not found."""
        text = "This text does not contain the expected patterns"

        result = classic_extractor._extract_field_value("customer_name", text)
        assert result is None

    def test_extract_field_value_regex_error(self, classic_extractor):
        """Test handling of regex errors during extraction."""
        with patch('src.pdf_parser.extractors.st') as mock_st:
            # Mock a regex that will cause an error
            mock_pattern = Mock()
            mock_pattern.search.side_effect = Exception("Regex error")
            classic_extractor.cfg["customer_name"] = [mock_pattern]

            result = classic_extractor._extract_field_value("customer_name", "test text")

            assert result is None
            mock_st.warning.assert_called_once()

    def test_extract_multiple_patterns_first_matches(self, test_config):
        """Test extraction with multiple patterns where first pattern matches."""
        config = {
            "multi_pattern": {
                "display": "Multi Pattern",
                "patterns": [
                    r"First Pattern[:\s]*([A-Za-z\s]+)",
                    r"Second Pattern[:\s]*([A-Za-z\s]+)"
                ]
            }
        }
        extractor = ClassicExtractor(config)
        text = "First Pattern: Match Found"

        result = extractor._extract_field_value("multi_pattern", text)
        assert result == "Match Found"


class TestAIExtractor:
    """Test cases for AIExtractor class."""

    def test_init_success(self):
        """Test successful initialization with API key."""
        with patch('src.pdf_parser.extractors.OpenAI') as mock_openai:
            extractor = AIExtractor("test-api-key")
            mock_openai.assert_called_once_with(api_key="test-api-key")

    def test_init_missing_api_key(self):
        """Test initialization with missing API key."""
        with pytest.raises(DataExtractionError, match="Missing OpenAI API key"):
            AIExtractor("")

        with pytest.raises(DataExtractionError, match="Missing OpenAI API key"):
            AIExtractor(None)

    def test_init_client_creation_error(self):
        """Test handling of OpenAI client creation errors."""
        with patch('src.pdf_parser.extractors.OpenAI') as mock_openai:
            mock_openai.side_effect = Exception("Client creation failed")

            with pytest.raises(DataExtractionError, match="OpenAI client initialization error"):
                AIExtractor("test-api-key")

    def test_chat_success(self, mock_openai_client):
        """Test successful OpenAI chat completion."""
        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")
            messages = [{"role": "user", "content": "test message"}]

            result = extractor._chat(messages)

            assert "customer_name" in result
            mock_openai_client.chat.completions.create.assert_called_once()

    def test_chat_empty_response(self, mock_openai_client):
        """Test handling of empty OpenAI response."""
        mock_openai_client.chat.completions.create.return_value.choices = []

        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")
            messages = [{"role": "user", "content": "test message"}]

            with pytest.raises(DataExtractionError, match="OpenAI returned empty response"):
                extractor._chat(messages)

    def test_chat_openai_error(self, mock_openai_client):
        """Test handling of OpenAI API errors."""
        from openai import OpenAIError
        mock_openai_client.chat.completions.create.side_effect = OpenAIError("API Error")

        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")
            messages = [{"role": "user", "content": "test message"}]

            with pytest.raises(DataExtractionError, match="OpenAI API error"):
                extractor._chat(messages)

    def test_discover_labels_success(self, mock_openai_client):
        """Test successful label discovery."""
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "Customer Name, Policy Number, Claim Amount"

        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")

            result = extractor.discover_labels("Sample PDF text content")

            assert len(result) == 3
            assert "Customer Name" in result
            assert "Policy Number" in result
            assert "Claim Amount" in result

    def test_discover_labels_empty_text(self, mock_openai_client):
        """Test label discovery with empty text."""
        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")

            with pytest.raises(DataExtractionError, match="No text content to analyze"):
                extractor.discover_labels("")

    def test_discover_labels_no_response(self, mock_openai_client):
        """Test label discovery with empty AI response."""
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = ""

        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")

            with pytest.raises(DataExtractionError, match="Unexpected error during AI call"):
                extractor.discover_labels("Sample text")

    def test_discover_labels_filters_invalid_labels(self, mock_openai_client):
        """Test that label discovery filters out invalid labels."""
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "A, AB, Valid Label Name, Another Valid Label, " + "x" * 50  # Too long label

        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")

            result = extractor.discover_labels("Sample text")

            # Should filter out labels that are too short or too long
            valid_labels = [label for label in result if 2 < len(label) < 40]
            assert len(result) == len(valid_labels)
            assert "Valid Label Name" in result
            assert "Another Valid Label" in result

    def test_extract_success(self, mock_openai_client):
        """Test successful AI data extraction."""
        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")
            fields = ["customer_name", "policy_number"]
            text = "Sample PDF content"

            result = extractor.extract(text, fields)

            assert "customer_name" in result
            assert "policy_number" in result
            assert result["customer_name"] == "John Doe"
            assert result["policy_number"] == "POL-123"

    def test_extract_empty_text(self, mock_openai_client):
        """Test extraction with empty text."""
        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")

            with pytest.raises(DataExtractionError, match="No text content to process"):
                extractor.extract("", ["field1"])

    def test_extract_no_fields(self, mock_openai_client):
        """Test extraction with no fields specified."""
        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")

            with pytest.raises(DataExtractionError, match="No fields specified for extraction"):
                extractor.extract("Sample text", [])

    def test_build_extraction_prompt(self, mock_openai_client):
        """Test extraction prompt building."""
        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")
            fields = ["field1", "field2"]
            text = "Sample text content"

            prompt = extractor._build_extraction_prompt(fields, text)

            assert "field1, field2" in prompt
            assert "compact JSON" in prompt
            assert text in prompt

    def test_build_extraction_prompt_long_text(self, mock_openai_client):
        """Test extraction prompt building with long text that gets truncated."""
        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")
            fields = ["field1"]
            text = "x" * 30000  # Longer than 20k limit

            prompt = extractor._build_extraction_prompt(fields, text)

            # Should be truncated to 20k characters plus the instruction text
            assert len(prompt) < 30000
            assert prompt.endswith("x" * 1000)  # Should end with the truncated text

    def test_parse_extraction_result_success(self, mock_openai_client):
        """Test successful parsing of extraction results."""
        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")

            response = 'Here is the extracted data: {"name": "John", "age": "30"}'
            result = extractor._parse_extraction_result(response)

            assert result == {"name": "John", "age": "30"}

    def test_parse_extraction_result_no_json(self, mock_openai_client):
        """Test parsing when no JSON is found in response."""
        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")

            response = "No JSON found in this response"

            with pytest.raises(DataExtractionError, match="AI did not return valid JSON"):
                extractor._parse_extraction_result(response)

    def test_parse_extraction_result_invalid_json(self, mock_openai_client):
        """Test parsing with invalid JSON format."""
        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")

            response = '{"name": "John", "incomplete":'

            with pytest.raises(DataExtractionError, match="AI did not return valid JSON"):
                extractor._parse_extraction_result(response)

    def test_parse_extraction_result_non_dict_json(self, mock_openai_client):
        """Test parsing when JSON is not a dictionary."""
        with patch('src.pdf_parser.extractors.OpenAI', return_value=mock_openai_client):
            extractor = AIExtractor("test-api-key")

            response = '["not", "a", "dictionary"]'

            with pytest.raises(DataExtractionError, match="AI did not return valid JSON"):
                extractor._parse_extraction_result(response)
