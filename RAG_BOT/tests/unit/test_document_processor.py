import unittest
from RAG_BOT.document_processor import DocumentProcessor
from langchain_core.documents import Document
from datetime import datetime

class TestDocumentProcessor(unittest.TestCase):

    def setUp(self):
        """Setup a DocumentProcessor instance for use in tests."""
        self.processor = DocumentProcessor()

    # Tests for extract_date_from_text (English formats)
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

    # Tests for extract_date_from_text (Hindi formats)
    def test_extract_date_from_text_hindi_yyyy_mm_dd_dash(self):
        text = "दिनांक: २०२३-१०-२७"
        self.assertEqual(self.processor.extract_date_from_text(text), "2023-10-27")

    def test_extract_date_from_text_hindi_dd_mm_yyyy_slash(self):
        text = "दिनांक: १५/०४/२०२३"
        self.assertEqual(self.processor.extract_date_from_text(text), "2023-04-15")

    def test_extract_date_from_text_hindi_dd_mm_yyyy_dot(self):
        text = "दिनांक: ०१.०१.२०२४"
        self.assertEqual(self.processor.extract_date_from_text(text), "2024-01-01")

    def test_extract_date_from_text_hindi_dd_mm_yy_dot(self):
        text = "दिनांक: १०.१२.२३"
        self.assertEqual(self.processor.extract_date_from_text(text), "2023-12-10")

    def test_extract_date_from_text_hindi_dd_mm_yy_dash(self):
        text = "दिनांक: ०५-०६-२२"
        self.assertEqual(self.processor.extract_date_from_text(text), "2022-06-05")

    def test_extract_date_from_text_hindi_d_m_yyyy_dot(self):
        text = "दिनांक: १.१.२०२४"
        self.assertEqual(self.processor.extract_date_from_text(text), "2024-01-01")

    def test_extract_date_from_text_hindi_dd_m_yyyy_dot(self):
        text = "दिनांक: १०.१.२०२४"
        self.assertEqual(self.processor.extract_date_from_text(text), "2024-01-10")

    def test_extract_date_from_text_hindi_d_mm_yyyy_dot(self):
        text = "दिनांक: १.१२.२०२४"
        self.assertEqual(self.processor.extract_date_from_text(text), "2024-12-01")

    # Tests for get_murli_type
    def test_get_murli_type_avyakt_english(self):
        text = "This is an Avyakt Murli."
        self.assertTrue(self.processor.get_murli_type(text))

    def test_get_murli_type_avyakt_english_case_insensitive(self):
        text = "This is an AVYAKT Murli."
        self.assertTrue(self.processor.get_murli_type(text))

    def test_get_murli_type_avyakt_hindi(self):
        text = "यह एक अव्यक्त मुरली है।"
        self.assertTrue(self.processor.get_murli_type(text))

    def test_get_murli_type_sakar_english(self):
        text = "This is a Sakar Murli."
        self.assertFalse(self.processor.get_murli_type(text))

    def test_get_murli_type_sakar_hindi(self):
        text = "यह एक साकार मुरली है।"
        self.assertFalse(self.processor.get_murli_type(text))


if __name__ == '__main__':
    unittest.main()
