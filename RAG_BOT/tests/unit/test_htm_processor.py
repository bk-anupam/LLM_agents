import unittest
from unittest.mock import patch, mock_open, MagicMock
import os

from RAG_BOT.htm_processor import HtmProcessor
from langchain_core.documents import Document

class TestHtmProcessor(unittest.TestCase):

    def setUp(self):
        """Setup an HtmProcessor instance for use in tests."""
        self.processor = HtmProcessor()

    @patch('RAG_BOT.htm_processor.os.path.basename')
    @patch('builtins.open', new_callable=mock_open)
    @patch('RAG_BOT.htm_processor.logger')
    def test_load_htm_success_sakar_murli(self, mock_logger, mock_file_open, mock_basename):
        """Test successful loading of an HTM file representing a Sakar Murli."""
        htm_content = "<html><body>Date: 01.01.2024 Some Sakar Murli content.</body></html>"
        mock_file_open.return_value.read.return_value = htm_content
        mock_basename.return_value = "test.htm"

        # Mock inherited methods
        with patch.object(self.processor, 'extract_date_from_text', return_value="2024-01-01") as mock_extract_date, \
             patch.object(self.processor, 'get_murli_type', return_value=False) as mock_get_type:

            doc = self.processor.load_htm("dummy/path/test.htm")

            self.assertIsNotNone(doc)
            self.assertIsInstance(doc, Document)
            self.assertEqual(doc.page_content, "Date: 01.01.2024 Some Sakar Murli content.")
            self.assertEqual(doc.metadata.get("source"), "test.htm")
            self.assertEqual(doc.metadata.get("full_path"), "dummy/path/test.htm")
            self.assertEqual(doc.metadata.get("date"), "2024-01-01")
            self.assertNotIn("is_avyakt", doc.metadata)
            mock_extract_date.assert_called_once()
            mock_get_type.assert_called_once()
            mock_logger.info.assert_called_with("Processed HTM: test.htm, Date: 2024-01-01, Is Avyakt: False")

    @patch('RAG_BOT.htm_processor.os.path.basename')
    @patch('builtins.open', new_callable=mock_open)
    @patch('RAG_BOT.htm_processor.logger')
    def test_load_htm_success_avyakt_murli(self, mock_logger, mock_file_open, mock_basename):
        """Test successful loading of an HTM file representing an Avyakt Murli."""
        htm_content = "<html><body>Date: 02.02.2024 Avyakt Murli content.</body></html>"
        mock_file_open.return_value.read.return_value = htm_content
        mock_basename.return_value = "avyakt_test.htm"

        with patch.object(self.processor, 'extract_date_from_text', return_value="2024-02-02") as mock_extract_date, \
             patch.object(self.processor, 'get_murli_type', return_value=True) as mock_get_type:

            doc = self.processor.load_htm("dummy/path/avyakt_test.htm")

            self.assertIsNotNone(doc)
            self.assertEqual(doc.page_content, "Date: 02.02.2024 Avyakt Murli content.")
            self.assertEqual(doc.metadata.get("source"), "avyakt_test.htm")
            self.assertEqual(doc.metadata.get("date"), "2024-02-02")
            self.assertTrue(doc.metadata.get("is_avyakt"))
            mock_logger.info.assert_called_with("Processed HTM: avyakt_test.htm, Date: 2024-02-02, Is Avyakt: True")

    @patch('RAG_BOT.htm_processor.os.path.basename')
    @patch('builtins.open', new_callable=mock_open)
    @patch('RAG_BOT.htm_processor.logger')
    def test_load_htm_encoding_windows1252(self, mock_logger, mock_file_open, mock_basename):
        """Test HTM loading with windows-1252 encoding."""
        htm_content_bytes = "<html><body>Date: 03.03.2024 Content with special chars like ©</body></html>".encode('windows-1252')
        
        # Simulate successful read with windows-1252
        mock_file_open.return_value.read.return_value = htm_content_bytes.decode('windows-1252')
        mock_basename.return_value = "win1252.htm"

        with patch.object(self.processor, 'extract_date_from_text', return_value="2024-03-03"), \
             patch.object(self.processor, 'get_murli_type', return_value=False):
            
            doc = self.processor.load_htm("dummy/path/win1252.htm")
            self.assertIsNotNone(doc)
            self.assertIn("Content with special chars like ©", doc.page_content)
            mock_file_open.assert_called_with("dummy/path/win1252.htm", 'r', encoding='windows-1252')

    @patch('RAG_BOT.htm_processor.os.path.basename')
    @patch('builtins.open', new_callable=mock_open)
    @patch('RAG_BOT.htm_processor.logger')
    def test_load_htm_encoding_utf8_fallback(self, mock_logger, mock_file_open, mock_basename):
        """Test HTM loading with UTF-8 fallback after windows-1252 fails."""
        htm_content_utf8 = "<html><body>Date: 04.04.2024 UTF-8 content €</body></html>"

        # Mock for the first file object (returned by the first open() call for windows-1252)
        mock_file_obj_win1252_fail = MagicMock(name="file_obj_win1252_fail")
        mock_file_obj_win1252_fail.__enter__.return_value = mock_file_obj_win1252_fail
        mock_file_obj_win1252_fail.read.side_effect = UnicodeDecodeError('windows-1252', b'', 0, 1, 'reason')
        mock_file_obj_win1252_fail.__exit__ = MagicMock(return_value=None)

        # Mock for the second file object (returned by the second open() call for utf-8)
        mock_file_obj_utf8_ok = MagicMock(name="file_obj_utf8_ok")
        mock_file_obj_utf8_ok.__enter__.return_value = mock_file_obj_utf8_ok
        mock_file_obj_utf8_ok.read.return_value = htm_content_utf8
        mock_file_obj_utf8_ok.__exit__ = MagicMock(return_value=None)

        # Update the mock_file_open side_effect to use kwargs
        def mock_open_with_encoding(*args, **kwargs):
            if kwargs.get('encoding') == 'windows-1252':
                return mock_file_obj_win1252_fail
            return mock_file_obj_utf8_ok
            
        mock_file_open.side_effect = mock_open_with_encoding
        mock_basename.return_value = "utf8_fallback.htm"

        with patch.object(self.processor, 'extract_date_from_text', return_value="2024-04-04"), \
             patch.object(self.processor, 'get_murli_type', return_value=False):

            doc = self.processor.load_htm("dummy/path/utf8_fallback.htm")
            self.assertIsNotNone(doc)
            self.assertIn("UTF-8 content €", doc.page_content)
            
            # Update the assertion to check kwargs instead of positional args
            first_call = mock_file_open.call_args_list[0]
            self.assertEqual(first_call.kwargs.get('encoding'), 'windows-1252')
            second_call = mock_file_open.call_args_list[1]
            self.assertEqual(second_call.kwargs.get('encoding'), 'utf-8')

    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    @patch('RAG_BOT.htm_processor.logger')
    def test_load_htm_file_not_found(self, mock_logger, mock_file_open):
        """Test HTM loading when the file is not found."""
        doc = self.processor.load_htm("dummy/path/nonexistent.htm")
        self.assertIsNone(doc)
        mock_logger.error.assert_called_with("HTM file not found: dummy/path/nonexistent.htm")

    @patch('builtins.open', new_callable=mock_open)
    @patch('RAG_BOT.htm_processor.logger')
    def test_load_htm_read_error_fallback_fails(self, mock_logger, mock_file_open):
        """Test HTM loading when both encoding reads fail."""
        # Mock for the first file object (windows-1252 read fails)
        mock_file_obj_win1252_fail = MagicMock(name="file_obj_win1252_fail_again")
        mock_file_obj_win1252_fail.__enter__.return_value = mock_file_obj_win1252_fail
        mock_file_obj_win1252_fail.read.side_effect = UnicodeDecodeError('windows-1252', b'', 0, 1, 'reason')
        mock_file_obj_win1252_fail.__exit__ = MagicMock(return_value=None)

        # Mock for the second file object (utf-8 read also fails)
        mock_file_obj_utf8_fail = MagicMock(name="file_obj_utf8_fail_again")
        mock_file_obj_utf8_fail.__enter__.return_value = mock_file_obj_utf8_fail
        mock_file_obj_utf8_fail.read.side_effect = Exception("UTF-8 read failed")
        mock_file_obj_utf8_fail.__exit__ = MagicMock(return_value=None)

        mock_file_open.side_effect = [
            mock_file_obj_win1252_fail,
            mock_file_obj_utf8_fail
        ]
        doc = self.processor.load_htm("dummy/path/read_error.htm")
        self.assertIsNone(doc)
        mock_logger.error.assert_called_with("Failed to read HTM file with utf-8 fallback: dummy/path/read_error.htm. Error: UTF-8 read failed")
        self.assertEqual(mock_file_open.call_count, 2)
        
    @patch('RAG_BOT.htm_processor.os.path.basename')
    @patch('builtins.open', new_callable=mock_open)
    @patch('RAG_BOT.htm_processor.BeautifulSoup', side_effect=Exception("Parsing failed"))
    @patch('RAG_BOT.htm_processor.logger')
    def test_load_htm_parse_error(self, mock_logger, mock_bs_constructor, mock_file_open, mock_basename):
        """Test HTM loading when BeautifulSoup parsing fails."""
        mock_file_open.return_value.read.return_value = "<html>bad html"
        mock_basename.return_value = "parse_error.htm"
        
        doc = self.processor.load_htm("dummy/path/parse_error.htm")
        self.assertIsNone(doc)
        mock_logger.error.assert_called_with("Failed to parse HTM or extract data from dummy/path/parse_error.htm. Error: Parsing failed")

    @patch('RAG_BOT.htm_processor.os.path.basename')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_htm_no_body_tag(self, mock_file_open, mock_basename):
        """Test HTM loading for a file with no body tag."""
        htm_content = "<html><head><title>No Body</title></head></html>"
        mock_file_open.return_value.read.return_value = htm_content
        mock_basename.return_value = "no_body.htm"

        with patch.object(self.processor, 'extract_date_from_text', return_value=None), \
             patch.object(self.processor, 'get_murli_type', return_value=None):
            doc = self.processor.load_htm("dummy/path/no_body.htm")
            self.assertIsNotNone(doc)
            self.assertEqual(doc.page_content, "") # Empty string as no body text
            self.assertNotIn("date", doc.metadata)
            self.assertNotIn("is_avyakt", doc.metadata)

    @patch('RAG_BOT.htm_processor.os.path.basename')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_htm_no_date_or_type_info(self, mock_file_open, mock_basename):
        """Test HTM loading when no date or murli type info is found."""
        htm_content = "<html><body>Just some plain text.</body></html>"
        mock_file_open.return_value.read.return_value = htm_content
        mock_basename.return_value = "no_info.htm"

        with patch.object(self.processor, 'extract_date_from_text', return_value=None) as mock_extract_date, \
             patch.object(self.processor, 'get_murli_type', return_value=None) as mock_get_type:
            doc = self.processor.load_htm("dummy/path/no_info.htm")
            self.assertIsNotNone(doc)
            self.assertEqual(doc.page_content, "Just some plain text.")
            self.assertNotIn("date", doc.metadata)
            self.assertNotIn("is_avyakt", doc.metadata)
            mock_extract_date.assert_called_once()
            mock_get_type.assert_called_once()

    @patch('RAG_BOT.htm_processor.os.path.basename')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_htm_whitespace_cleaning(self, mock_file_open, mock_basename):
        """Test that whitespace is cleaned from the HTM body content."""
        htm_content = "<html><body>   Extra   spaces\nand\nnewlines.   </body></html>"
        mock_file_open.return_value.read.return_value = htm_content
        mock_basename.return_value = "whitespace.htm"

        with patch.object(self.processor, 'extract_date_from_text', return_value=None), \
             patch.object(self.processor, 'get_murli_type', return_value=None):
            doc = self.processor.load_htm("dummy/path/whitespace.htm")
            self.assertIsNotNone(doc)
            self.assertEqual(doc.page_content, "Extra spaces and newlines.")

    @patch('RAG_BOT.htm_processor.os.path.isdir')
    @patch('RAG_BOT.htm_processor.os.listdir')
    @patch.object(HtmProcessor, 'load_htm') # Patching the method on the class
    @patch('RAG_BOT.htm_processor.logger')
    def test_load_directory_htm_success(self, mock_logger, mock_load_htm_method, mock_listdir, mock_isdir):
        """Test loading HTM files from a directory successfully."""
        mock_isdir.return_value = True
        mock_listdir.return_value = ["file1.htm", "file2.html", "notes.txt", "file3.HTM"]
        
        doc1 = Document(page_content="content1", metadata={"source": "file1.htm"})
        doc2 = Document(page_content="content2", metadata={"source": "file2.html"})
        doc3 = Document(page_content="content3", metadata={"source": "file3.HTM"})
        
        # Configure load_htm to return different docs for different files
        def side_effect_load_htm(file_path):
            if file_path.endswith("file1.htm"): return doc1
            if file_path.endswith("file2.html"): return doc2
            if file_path.endswith("file3.HTM"): return doc3
            return None
        mock_load_htm_method.side_effect = side_effect_load_htm

        docs = self.processor.load_directory_htm("dummy_dir")

        self.assertEqual(len(docs), 3)
        self.assertIn(doc1, docs)
        self.assertIn(doc2, docs)
        self.assertIn(doc3, docs)
        self.assertEqual(mock_load_htm_method.call_count, 3)
        # Check calls to load_htm (os.path.join will construct the full path)
        expected_calls = [
            unittest.mock.call(os.path.join("dummy_dir", "file1.htm")),
            unittest.mock.call(os.path.join("dummy_dir", "file2.html")),
            unittest.mock.call(os.path.join("dummy_dir", "file3.HTM"))
        ]
        mock_load_htm_method.assert_has_calls(expected_calls, any_order=True)
        mock_logger.info.assert_any_call("Scanning directory for HTM files: dummy_dir")
        mock_logger.info.assert_any_call("Found 3 HTM/HTML files. Successfully processed and loaded 3 documents from dummy_dir")

    @patch('RAG_BOT.htm_processor.os.path.isdir', return_value=True)
    @patch('RAG_BOT.htm_processor.os.listdir', return_value=[])
    @patch('RAG_BOT.htm_processor.logger')
    def test_load_directory_htm_empty(self, mock_logger, mock_listdir, mock_isdir):
        """Test loading from an empty directory."""
        docs = self.processor.load_directory_htm("empty_dir")
        self.assertEqual(len(docs), 0)
        mock_logger.info.assert_any_call("Found 0 HTM/HTML files. Successfully processed and loaded 0 documents from empty_dir")

    @patch('RAG_BOT.htm_processor.os.path.isdir', return_value=False)
    @patch('RAG_BOT.htm_processor.logger')
    def test_load_directory_htm_non_existent(self, mock_logger, mock_isdir):
        """Test loading from a non-existent directory."""
        docs = self.processor.load_directory_htm("non_existent_dir")
        self.assertEqual(len(docs), 0)
        mock_logger.error.assert_called_with("Directory not found: non_existent_dir")

    @patch('RAG_BOT.htm_processor.os.path.isdir', return_value=True)
    @patch('RAG_BOT.htm_processor.os.listdir')
    @patch.object(HtmProcessor, 'load_htm')
    @patch('RAG_BOT.htm_processor.logger')
    def test_load_directory_htm_one_file_fails(self, mock_logger, mock_load_htm_method, mock_listdir, mock_isdir):
        """Test directory loading where one HTM file processing fails."""
        mock_listdir.return_value = ["good.htm", "bad.htm", "good_again.html"]
        doc1 = Document(page_content="good1", metadata={"source": "good.htm"})
        doc3 = Document(page_content="good3", metadata={"source": "good_again.html"})

        def side_effect_load_htm(file_path):
            if file_path.endswith("good.htm"): return doc1
            if file_path.endswith("bad.htm"): return None # Simulate failure
            if file_path.endswith("good_again.html"): return doc3
            return None
        mock_load_htm_method.side_effect = side_effect_load_htm

        docs = self.processor.load_directory_htm("mixed_dir")
        self.assertEqual(len(docs), 2)
        self.assertIn(doc1, docs)
        self.assertIn(doc3, docs)
        mock_logger.warning.assert_called_with("Skipped processing file: bad.htm")
        mock_logger.info.assert_any_call("Found 3 HTM/HTML files. Successfully processed and loaded 2 documents from mixed_dir")

if __name__ == '__main__':
    unittest.main()