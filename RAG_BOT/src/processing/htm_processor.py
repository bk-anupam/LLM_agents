import os
import re
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from RAG_BOT.src.logger import logger
from RAG_BOT.src.processing.document_processor import DocumentProcessor 


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
            # Try reading with windows-1252 first, as it's common for older HTM
            with open(htm_path, 'r', encoding='windows-1252') as f:
                html_content = f.read()
        except UnicodeDecodeError:
            # Fallback to utf-8 if windows-1252 fails
            try:
                with open(htm_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read HTM file with utf-8 fallback: {htm_path}. Error: {e}")
                return None
        except FileNotFoundError:
             logger.error(f"HTM file not found: {htm_path}")
             return None
        except Exception as e:
            logger.error(f"Failed to read HTM file: {htm_path}. Error: {e}")
            return None

        try:
            soup = BeautifulSoup(html_content, 'lxml') # Use lxml parser for robustness
            # Extract text from the body tag, handling cases where body might be missing
            body_tag = soup.find('body')
            body_text = body_tag.get_text(separator='\n', strip=True) if body_tag else ''

            # Improve text cleaning: replace multiple whitespace chars (including newlines) with a single space
            body_text = re.sub(r'\s+', ' ', body_text).strip()

            # Extract metadata using base class methods
            # Check a reasonable portion of the text for metadata clues
            check_text = body_text[:500]
            date = self.extract_date_from_text(check_text)
            is_avyakt = self.get_murli_type(check_text)

            # Extract filename from the path for the source metadata
            filename = os.path.basename(htm_path)

            metadata = {
                "source": filename, # Use filename instead of full path
                "full_path": htm_path # Optionally keep the full path if needed elsewhere
            }
            if date:
                metadata["date"] = date
            if is_avyakt is True: # Explicit check for True
                metadata["is_avyakt"] = True

            # Log less verbose debug message unless needed
            # logger.debug(f"Extracted content from HTM {filename}: {body_text[:200]}...") # Log preview
            logger.info(f"Processed HTM: {filename}, Date: {date}, Is Avyakt: {is_avyakt}")

            return Document(page_content=body_text, metadata=metadata)

        except Exception as e:
            # Catch potential errors during parsing or metadata extraction
            logger.error(f"Failed to parse HTM or extract data from {htm_path}. Error: {e}")
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

        logger.info(f"Scanning directory for HTM files: {directory_path}")
        file_count = 0
        processed_count = 0
        for filename in os.listdir(directory_path):
            # Check for both .htm and .html extensions, case-insensitive
            if filename.lower().endswith((".htm", ".html")):
                file_count += 1
                htm_path = os.path.join(directory_path, filename)
                logger.debug(f"Attempting to load HTM file: {htm_path}")
                document = self.load_htm(htm_path) # Use self.load_htm
                if document:
                    all_documents.append(document)
                    processed_count += 1
                else:
                    logger.warning(f"Skipped processing file: {filename}")


        logger.info(f"Found {file_count} HTM/HTML files. Successfully processed and loaded {processed_count} documents from {directory_path}")
        return all_documents


if __name__ == "__main__":
    # Example usage with the new class structure
    # Use a relative path for better portability
    script_dir = os.path.dirname(__file__)
    TEST_DATA_DIR = os.path.join(script_dir, 'tests', 'data', 'hindi')

    # Instantiate the processor
    htm_processor = HtmProcessor()

    # Load documents using the instance method
    print(f"Loading HTM documents from: {TEST_DATA_DIR}")
    htm_documents = htm_processor.load_directory_htm(TEST_DATA_DIR)

    if htm_documents:
        print(f"\nSuccessfully loaded {len(htm_documents)} HTM documents.")        
        # Print metadata of first few docs to verify loading
        print("\n--- Sample Document Metadata and Content ---")
        for i in range(min(5, len(htm_documents))):
            print(f"\nDocument {i}:")
            print(f"  Metadata: {htm_documents[i].metadata}")
            # Limit content preview length
            content_preview = htm_documents[i].page_content[:300].replace('\n', ' ') + "..."
            print(f"  Content Preview: {content_preview}")
    else:
        print(f"No HTM documents were successfully processed from {TEST_DATA_DIR}")

