# /home/bk_anupam/code/LLM_agents/RAG_BOT/file_manager.py
import os
import shutil
from RAG_BOT.logger import logger
from RAG_BOT.config import Config
from typing import Optional

class FileManager:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def move_indexed_file(self, source_path: str, language: str) -> bool:
        """
        Moves a successfully indexed file to the 'indexed_data' directory,
        replicating the exact directory structure from the original source path.
        For example:
            source_path = RAG_BOT/data/hindi/1969/filename.pdf
            indexed_data_path = RAG_BOT/indexed_data
            Will result in a destination of: RAG_BOT/indexed_data/hindi/1969/filename.pdf
        """
        if self.config.INDEXED_DATA_PATH is None:
            logger.error("INDEXED_DATA_PATH not configured. Cannot move indexed files.")
            return False

        if not source_path or not os.path.exists(source_path):
            logger.warning(f"Source file path invalid or file does not exist: {source_path}. Cannot move.")
            return False

        try:
            file_name = os.path.basename(source_path)
            # Define the root folder for source files
            source_data_root = os.path.join("RAG_BOT", "data")
            # Get the directory of the source file and extract subdirectories relative to source_data_root
            source_dir = os.path.dirname(source_path)
            relative_subdirs = os.path.relpath(source_dir, start=source_data_root)
            # If no subdirectories are found, relative_subdirs might be '.'; in that case, ignore it
            if relative_subdirs == '.':
                relative_subdirs = ''
            # Build the destination directory path by concatenating the indexed_data path and the subdirectories
            destination_dir = os.path.join(self.config.INDEXED_DATA_PATH, relative_subdirs)
            # Normalize the destination directory to remove any trailing slashes
            destination_dir = os.path.normpath(destination_dir)
            destination_path = os.path.join(destination_dir, file_name)

            # Ensure that the destination directory exists
            os.makedirs(destination_dir, exist_ok=True)

            shutil.move(source_path, destination_path)
            logger.info(f"Successfully moved indexed file '{file_name}' to: {destination_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to move file {source_path} to destination: {e}", exc_info=True)
            return False
