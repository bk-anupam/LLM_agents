import os
import re
import sys
from bs4 import BeautifulSoup
from langchain_core.documents import Document

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from RAG_BOT.logger import logger
# Remove the old import: from RAG_BOT.pdf_processor import extract_date_from_text, get_murli_type
from RAG_BOT.document_processor import DocumentProcessor # Import the base class


class HtmProcessor(DocumentProcessor):
    """
    Processes HTM files to extract text and metadata, inheriting common
    functionality from DocumentProcessor.
    """
    def load_htm(self, htm_path):
        """
        Loads an HTM file, extracts text content and metadata.

        Args:
            htm_path (str): Path to the HTM file.

        Returns:
            Document or None: A Document object with extracted content and metadata, or None if processing fails.
        """
        try:
            with open(htm_path, 'r', encoding='windows-1252') as f:
                html_content = f.read()
        except UnicodeDecodeError:
            try:
                with open(htm_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read HTM file with utf-8: {htm_path}. Error: {e}")
                return None
        except Exception as e:
            logger.error(f"Failed to read HTM file: {htm_path}. Error: {e}")
            return None

        try:
            soup = BeautifulSoup(html_content, 'lxml') # Use lxml parser
            # Extract text from the body tag
            body_text = soup.body.get_text(separator='\n', strip=True) if soup.body else ''
            # Collapse multiple newlines and trim whitespace            
            body_text = re.sub(r'\n+', ' ', body_text).strip()
            # Extract metadata using base class methods
            date = self.extract_date_from_text(body_text[:500]) # Check first 500 chars for date
            is_avyakt = self.get_murli_type(body_text[:500]) # Check first 500 chars for type

            metadata = {
                "source": htm_path,
            }
            if date:
                metadata["date"] = date
            if is_avyakt is True:
                metadata["is_avyakt"] = True
            logger.debug(f"Extracted content from HTM {htm_path}: {body_text}")
            logger.info(f"Processed HTM: {htm_path}, Date: {date}, Is Avyakt: {is_avyakt}")

            return Document(page_content=body_text, metadata=metadata)

        except Exception as e:
            logger.error(f"Failed to parse HTM or extract data: {htm_path}. Error: {e}")
            return None


    def load_directory_htm(self, directory_path):
        """
        Loads all HTM files from a directory and processes them.

        Args:
            directory_path (str): Path to the directory containing HTM files.

        Returns:
            list[Document]: A list of Document objects from the processed HTM files.
        """
        all_documents = []
        if not os.path.isdir(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return []

        for filename in os.listdir(directory_path):
            if filename.endswith(".htm"):
                htm_path = os.path.join(directory_path, filename)
                document = self.load_htm(htm_path) # Use self.load_htm
                if document:
                    all_documents.append(document)

        logger.info(f"Finished loading {len(all_documents)} HTM documents from {directory_path}")
        return all_documents


if __name__ == "__main__":
    # Example usage with the new class structure
    TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'tests', 'data', 'hindi')
    # Instantiate the processor
    htm_processor = HtmProcessor()
    # Load documents using the instance method
    htm_documents = htm_processor.load_directory_htm(TEST_DATA_DIR)
    if htm_documents:
        # Optional: Split the documents using inherited methods
        # chunks = htm_processor.split_text(htm_documents)
        # chunks = htm_processor.semantic_chunking(htm_documents)
        # print(f"Total chunks created: {len(chunks)}")

        # Print metadata of first few docs to verify loading
        for i in range(min(5, len(htm_documents))):
            print(f"Doc {i} Metadata: {htm_documents[i].metadata}")
            print(f"Doc {i} Content Preview: {htm_documents[i].page_content[:500]}...")
    else:
        print(f"No HTM documents processed from {TEST_DATA_DIR}")
