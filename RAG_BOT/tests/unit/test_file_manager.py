import os
import shutil
import unittest
from unittest.mock import patch, MagicMock
from RAG_BOT.src.file_manager import FileManager

class DummyConfig:
    def __init__(self, indexed_data_path):
        self.INDEXED_DATA_PATH = indexed_data_path

class TestFileManager(unittest.TestCase):
    def setUp(self):
        self.dummy_indexed_path = "/tmp/indexed_data"
        self.file_manager = FileManager()
        # Override the config with a dummy config for testing
        self.file_manager.config = DummyConfig(self.dummy_indexed_path)

    @patch("RAG_BOT.file_manager.shutil.move")
    @patch("RAG_BOT.file_manager.os.makedirs")
    @patch("RAG_BOT.file_manager.os.path.exists", return_value=True)
    def test_move_indexed_file_success(self, mock_exists, mock_makedirs, mock_shutil_move):
        # Source with subdirectories under RAG_BOT/data
        source_path = os.path.join("RAG_BOT", "data", "hindi", "1969", "filename.pdf")
        # Expected relative subdirectory: hindi/1969
        expected_dir = os.path.join(self.dummy_indexed_path, "hindi", "1969")
        expected_destination = os.path.join(expected_dir, "filename.pdf")
        
        result = self.file_manager.move_indexed_file(source_path, "hindi")
        
        mock_exists.assert_called_with(source_path)
        mock_makedirs.assert_called_with(expected_dir, exist_ok=True)
        mock_shutil_move.assert_called_with(source_path, expected_destination)
        self.assertTrue(result)

    @patch("RAG_BOT.file_manager.os.path.exists", return_value=False)
    def test_move_indexed_file_nonexistent_source(self, mock_exists):
        source_path = os.path.join("RAG_BOT", "data", "english", "file.txt")
        result = self.file_manager.move_indexed_file(source_path, "english")
        self.assertFalse(result)

    @patch("RAG_BOT.file_manager.os.path.exists", return_value=True)
    def test_move_indexed_file_no_indexed_path_configured(self, mock_exists):
        # Set the indexed path to None
        self.file_manager.config.INDEXED_DATA_PATH = None
        source_path = os.path.join("RAG_BOT", "data", "english", "file.txt")
        result = self.file_manager.move_indexed_file(source_path, "english")
        self.assertFalse(result)

    @patch("RAG_BOT.file_manager.shutil.move")
    @patch("RAG_BOT.file_manager.os.makedirs")
    @patch("RAG_BOT.file_manager.os.path.exists", return_value=True)
    def test_move_indexed_file_no_subdirectory(self, mock_exists, mock_makedirs, mock_shutil_move):
        # Source file directly under RAG_BOT/data with no subdirectory.
        source_path = os.path.join("RAG_BOT", "data", "file.pdf")
        expected_dir = self.dummy_indexed_path  # relative_subdirs will be empty
        expected_destination = os.path.join(expected_dir, "file.pdf")
        
        result = self.file_manager.move_indexed_file(source_path, "")
        
        mock_exists.assert_called_with(source_path)
        mock_makedirs.assert_called_with(expected_dir, exist_ok=True)
        mock_shutil_move.assert_called_with(source_path, expected_destination)
        self.assertTrue(result)

if __name__ == "__main__":
    unittest.main()