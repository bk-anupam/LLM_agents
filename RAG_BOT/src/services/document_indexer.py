import os
from RAG_BOT.src.logger import logger
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.processing.pdf_processor import PdfProcessor
from RAG_BOT.src.processing.htm_processor import HtmProcessor
from RAG_BOT.src.persistence.vector_store import VectorStore
from RAG_BOT.src.file_manager import FileManager
from typing import Optional

class DocumentIndexer:
    def __init__(self, vector_store_instance: VectorStore, file_manager_instance: FileManager, 
                 config: Optional[Config] = None):
        self.config = config or Config()
        self.pdf_processor = PdfProcessor()
        self.htm_processor = HtmProcessor()
        self.vector_store = vector_store_instance
        self.file_manager = file_manager_instance

    def index_directory(self, base_data_path: str = None, move_indexed_files: bool = True):
        """
        Recursively finds PDF files in 'english' subdirectory and HTM files in 'hindi'
        subdirectory of the base_data_path, adds language metadata, and indexes them,
        delegating to VectorStore and using FileManager to move processed files.
        """
        if base_data_path is None:
            base_data_path = self.config.DATA_PATH # Use path from config if not provided
            if not base_data_path:
                logger.error("DATA_PATH not configured. Cannot index directory.")
                return

        logger.info(f"Starting multilingual indexing for base directory: {base_data_path}")
        if not os.path.isdir(base_data_path):
            logger.error(f"Provided base path is not a valid directory: {base_data_path}")
            return

        english_dir = os.path.join(base_data_path, 'english')
        hindi_dir = os.path.join(base_data_path, 'hindi')

        files_to_process = []
        # Process English PDFs
        if os.path.isdir(english_dir):
            logger.info(f"Scanning for English PDFs in: {english_dir}")
            for root, _, files in os.walk(english_dir):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        files_to_process.append({'path': os.path.join(root, file), 'lang': 'en'})
        else:
            logger.warning(f"English data directory not found: {english_dir}")

        # Process Hindi HTMs
        if os.path.isdir(hindi_dir):
            logger.info(f"Scanning for Hindi HTMs in: {hindi_dir}")
            for root, _, files in os.walk(hindi_dir):
                for file in files:
                    if file.lower().endswith(".htm"):
                         files_to_process.append({'path': os.path.join(root, file), 'lang': 'hi'})
        else:
            logger.warning(f"Hindi data directory not found: {hindi_dir}")

        if not files_to_process:
            logger.warning(f"No PDF or HTM files found in the respective subdirectories of: {base_data_path}")
            return

        logger.info(f"Found {len(files_to_process)} PDF/HTM files to potentially index.")
        indexed_count = 0
        skipped_count = 0
        moved_count = 0
        failed_count = 0

        for file_info in files_to_process:
            file_path = file_info['path']
            language = file_info['lang']
            try:
                documents = []
                if language == 'en' and file_path.lower().endswith(".pdf"):
                    documents = self.pdf_processor.load_pdf(file_path)
                elif language == 'hi' and file_path.lower().endswith(".htm"):
                    doc = self.htm_processor.load_htm(file_path)
                    if doc:
                        documents.append(doc) # load_htm returns a single doc

                if documents:
                    for doc in documents:
                        doc.metadata['language'] = language
                    logger.debug(f"Added language metadata '{language}' to document: {os.path.basename(file_path)}")
                else:
                    logger.warning(f"No documents loaded for file: {file_path}. Skipping indexing for this file.")
                    failed_count +=1 # Count as failure if no docs loaded
                    continue

                # Delegate to VectorStore's _index_document method
                # This method in VectorStore will handle checking existence, chunking, and adding to DB
                was_indexed = self.vector_store.index_document(
                    documents,
                    semantic_chunk=self.config.SEMANTIC_CHUNKING
                )

                if was_indexed and move_indexed_files:
                    indexed_count += 1
                    if self.file_manager.move_indexed_file(file_path, language):
                        moved_count += 1
                else:
                    # This counts skips due to already existing docs or failures within _index_document
                    skipped_count += 1 # or failed_count if _index_document returns specific error codes
            except Exception as e:
                logger.error(f"Unhandled exception during indexing attempt for {file_path}: {e}", exc_info=True)
                failed_count += 1

        logger.info(f"Directory indexing complete for: {base_data_path}")
        logger.info(f"Summary: Indexed={indexed_count}, Moved={moved_count}, Skipped/IndexFailed={skipped_count}, UnhandledFailures={failed_count}")
