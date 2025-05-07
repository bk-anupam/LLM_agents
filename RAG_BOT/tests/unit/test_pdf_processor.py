import unittest
from unittest.mock import patch, MagicMock
from RAG_BOT.pdf_processor import PdfProcessor # Import the class
from langchain_core.documents import Document
from datetime import datetime

class TestPDFProcessor(unittest.TestCase):

    def setUp(self):
        """Setup a PdfProcessor instance for use in tests."""
        self.processor = PdfProcessor()

    def test_extract_date_from_text_yyyy_mm_dd(self):
        text = "Document Date: 2023-10-27"
        self.assertEqual(self.processor.extract_date_from_text(text), "2023-10-27")

    def test_extract_date_from_text_dd_mm_yyyy_slash(self):
        text = "Date: 15/04/2023"
        self.assertEqual(self.processor.extract_date_from_text(text), "2023-04-15")

    def test_extract_date_from_text_dd_mm_yyyy_dot(self):
        text = "Date: 01.01.2024"
        self.assertEqual(self.processor.extract_date_from_text(text), "2024-01-01")

    def test_extract_date_from_text_dd_mm_yy_dot(self):
        text = "Date: 10.12.23"
        self.assertEqual(self.processor.extract_date_from_text(text), "2023-12-10")

    def test_extract_date_from_text_dd_mm_yy_dash(self):
        text = "Date: 05-06-22"
        self.assertEqual(self.processor.extract_date_from_text(text), "2022-06-05")

    def test_extract_date_from_text_d_m_yyyy_dot(self):
        text = "Date: 1.1.2024"
        self.assertEqual(self.processor.extract_date_from_text(text), "2024-01-01")

    def test_extract_date_from_text_dd_m_yyyy_dot(self):
        text = "Date: 10.1.2024"
        self.assertEqual(self.processor.extract_date_from_text(text), "2024-01-10")

    def test_extract_date_from_text_d_mm_yyyy_dot(self):
        text = "Date: 1.12.2024"
        self.assertEqual(self.processor.extract_date_from_text(text), "2024-12-01")

    def test_extract_date_from_text_no_date(self):
        text = "This text has no date."
        self.assertIsNone(self.processor.extract_date_from_text(text))

    def test_get_murli_type_avyakt(self):
        text = "This is an Avyakt Murli."
        self.assertTrue(self.processor.get_murli_type(text))

    def test_get_murli_type_avyakt_case_insensitive(self):
        text = "This is an AVYAKT Murli."
        self.assertTrue(self.processor.get_murli_type(text))

    def test_get_murli_type_sakar(self):
        text = "This is a Sakar Murli."
        self.assertFalse(self.processor.get_murli_type(text))

    @patch('RAG_BOT.pdf_processor.PyMuPDFLoader') # Changed to PyMuPDFLoader
    def test_load_pdf_multi_murli_single_murli(self, mock_pymupdfloader):
        # Mock PyPDFLoader to return pages for a single Murli
        mock_page1 = Document(page_content="Date: 01.01.2024\nContent of page 1.", metadata={"page": 0, "source": "test.pdf"})
        mock_page2 = Document(page_content="Content of page 2.", metadata={"page": 1, "source": "test.pdf"})
        mock_page3 = Document(page_content="Content of page 3.", metadata={"page": 2, "source": "test.pdf"})
        mock_pymupdfloader.return_value.load.return_value = [mock_page1, mock_page2, mock_page3]

        documents = self.processor.load_pdf("test.pdf")

        self.assertEqual(len(documents), 3)
        self.assertEqual(documents[0].metadata.get("date"), "2024-01-01")
        self.assertEqual(documents[1].metadata.get("date"), "2024-01-01")
        self.assertEqual(documents[2].metadata.get("date"), "2024-01-01")
        self.assertNotIn("is_avyakt", documents[0].metadata)
        self.assertNotIn("is_avyakt", documents[1].metadata)
        self.assertNotIn("is_avyakt", documents[2].metadata)


    @patch('RAG_BOT.pdf_processor.PyMuPDFLoader') # Changed to PyMuPDFLoader
    def test_load_pdf_multi_murli_multiple_murlis(self, mock_pymupdfloader):
        # Mock PyPDFLoader to return pages for multiple Murlis
        mock_page1 = Document(page_content="Date: 01.01.2024\nContent of Murli 1, page 1.", metadata={"page": 0, "source": "test.pdf"})
        mock_page2 = Document(page_content="Content of Murli 1, page 2.", metadata={"page": 1, "source": "test.pdf"})
        mock_page3 = Document(page_content="Date: 02.01.2024\nContent of Murli 2, page 1.", metadata={"page": 2, "source": "test.pdf"})
        mock_page4 = Document(page_content="Content of Murli 2, page 2.", metadata={"page": 3, "source": "test.pdf"})
        mock_pymupdfloader.return_value.load.return_value = [mock_page1, mock_page2, mock_page3, mock_page4]

        documents = self.processor.load_pdf("test.pdf")

        self.assertEqual(len(documents), 4)
        self.assertEqual(documents[0].metadata.get("date"), "2024-01-01")
        self.assertEqual(documents[1].metadata.get("date"), "2024-01-01")
        self.assertEqual(documents[2].metadata.get("date"), "2024-01-02")
        self.assertEqual(documents[3].metadata.get("date"), "2024-01-02")
        self.assertNotIn("is_avyakt", documents[0].metadata)
        self.assertNotIn("is_avyakt", documents[1].metadata)
        self.assertNotIn("is_avyakt", documents[2].metadata)
        self.assertNotIn("is_avyakt", documents[3].metadata)

    @patch('RAG_BOT.pdf_processor.PyMuPDFLoader') # Changed to PyMuPDFLoader
    def test_load_pdf_multi_murli_with_avyakt(self, mock_pymupdfloader):
        # Mock PyPDFLoader to return pages including an Avyakt Murli
        mock_page1 = Document(page_content="Date: 01.01.2024\nContent of Sakar Murli, page 1.", metadata={"page": 0, "source": "test.pdf"})
        mock_page2 = Document(page_content="Date: 02.01.2024 Avyakt Murli\nContent of Avyakt Murli, page 1.", metadata={"page": 1, "source": "test.pdf"})
        mock_page3 = Document(page_content="Content of Avyakt Murli, page 2.", metadata={"page": 2, "source": "test.pdf"})
        mock_pymupdfloader.return_value.load.return_value = [mock_page1, mock_page2, mock_page3]

        documents = self.processor.load_pdf("test.pdf")

        self.assertEqual(len(documents), 3)
        self.assertEqual(documents[0].metadata.get("date"), "2024-01-01")
        self.assertNotIn("is_avyakt", documents[0].metadata)
        self.assertEqual(documents[1].metadata.get("date"), "2024-01-02")
        self.assertTrue(documents[1].metadata.get("is_avyakt"))
        self.assertEqual(documents[2].metadata.get("date"), "2024-01-02")
        self.assertTrue(documents[2].metadata.get("is_avyakt"))

    @patch('RAG_BOT.pdf_processor.PyMuPDFLoader') # Changed to PyMuPDFLoader
    def test_load_pdf_multi_murli_no_date_or_avyakt(self, mock_pymupdfloader):
        # Mock PyPDFLoader to return pages with no date or avyakt
        mock_page1 = Document(page_content="Content of page 1.", metadata={"page": 0, "source": "test.pdf"})
        mock_page2 = Document(page_content="Content of page 2.", metadata={"page": 1, "source": "test.pdf"})
        mock_pymupdfloader.return_value.load.return_value = [mock_page1, mock_page2]

        documents = self.processor.load_pdf("test.pdf")

        self.assertEqual(len(documents), 2)
        self.assertNotIn("date", documents[0].metadata)
        self.assertNotIn("is_avyakt", documents[0].metadata)
        self.assertNotIn("date", documents[1].metadata)
        self.assertNotIn("is_avyakt", documents[1].metadata)
    
    @patch('RAG_BOT.pdf_processor.PyMuPDFLoader') # Changed to PyMuPDFLoader
    @patch('RAG_BOT.pdf_processor.logger') # Mock logger
    def test_load_pdf_failure(self, mock_logger, mock_pymupdfloader):
        # Mock PyMuPDFLoader to raise an exception
        mock_pymupdfloader.return_value.load.side_effect = Exception("Failed to load")
        
        documents = self.processor.load_pdf("test_fail.pdf")
        self.assertEqual(len(documents), 0)
        mock_logger.error.assert_called_with("Failed to load PDF: test_fail.pdf. Error: Failed to load")

    @patch('RAG_BOT.pdf_processor.PyMuPDFLoader') # Changed to PyMuPDFLoader
    def test_load_pdf_multi_murli_empty_pdf(self, mock_pymupdfloader):
        # Mock PyPDFLoader to return an empty list of pages
        mock_pymupdfloader.return_value.load.return_value = []

        documents = self.processor.load_pdf("test.pdf")

        self.assertEqual(len(documents), 0)

if __name__ == '__main__':
    unittest.main()
