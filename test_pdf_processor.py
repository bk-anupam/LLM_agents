import unittest
from unittest.mock import patch, mock_open
from pdf_processor import extract_date_from_text, load_pdf, Document
from datetime import datetime

class TestPDFProcessor(unittest.TestCase):

    def test_extract_date_from_text_yyyy_mm_dd(self):
        text = "Document Date: 2023-10-27"
        self.assertEqual(extract_date_from_text(text), "2023-10-27")
    
    def test_extract_date_from_text_dd_mm_yyyy_slash(self):
        text = "Date: 15/04/2023"
        self.assertEqual(extract_date_from_text(text), "2023-04-15")    

    @patch('pdf_processor.PyPDFLoader')
    def test_load_pdf_success(self, mock_pypdfloader):
        # Mock the PyPDFLoader to return a sample page
        mock_page_content = "Document Date: 2023-11-15"
        mock_metadata = {"page": 1, "source": "test.pdf"}
        mock_page = type('', (), {'page_content': mock_page_content, 'metadata': mock_metadata})()
        mock_pypdfloader.return_value.load_and_split.return_value = [mock_page]

        # Call the function
        documents = load_pdf("test.pdf")

        # Assertions
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, mock_page_content)
        self.assertEqual(documents[0].metadata["date"], "2023-11-15")
        self.assertEqual(documents[0].metadata["page"], 1)
        self.assertEqual(documents[0].metadata["source"], "test.pdf")

    @patch('pdf_processor.PyPDFLoader')
    def test_load_pdf_no_date(self, mock_pypdfloader):
        # Mock the PyPDFLoader to return a sample page with no date
        mock_page_content = "This document has no date"
        mock_metadata = {"page": 1, "source": "test.pdf"}
        mock_page = type('', (), {'page_content': mock_page_content, 'metadata': mock_metadata})()
        mock_pypdfloader.return_value.load_and_split.return_value = [mock_page]

        # Call the function
        documents = load_pdf("test.pdf")

        # Assertions
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, mock_page_content)
        self.assertNotIn("date", documents[0].metadata)
        self.assertEqual(documents[0].metadata["page"], 1)
        self.assertEqual(documents[0].metadata["source"], "test.pdf")

if __name__ == '__main__':
    unittest.main()
