import re
import os
import sys
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.documents import Document
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from RAG_BOT.logger import logger


class DocumentProcessor:
    """
    Base class for processing documents (PDF, HTM, etc.) to extract text,
    metadata, and split content into chunks.
    """

    def extract_date_from_text(self, text):
        """
        Attempts to extract a date from the given text and returns it in YYYY-MM-DD format.
        Args:
            text (str): The text to search for a date.
        Returns:
            str or None: The extracted date in YYYY-MM-DD format if found, otherwise None.
        """
        # Specific date patterns to avoid ambiguity
        date_patterns = [
            (r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d"),  # YYYY-MM-DD
            (r"(\d{2})/(\d{2})/(\d{4})", "%d/%m/%Y"), # DD/MM/YYYY
            (r"(\d{2})\.(\d{2})\.(\d{4})", "%d.%m.%Y"), # DD.MM.YYYY
            (r"(\d{1,2})\.(\d{1,2})\.(\d{4})", "%d.%m.%Y"), # D.M.YYYY, DD.M.YYYY, D.MM.YYYY
            (r"(\d{1,2})/(\d{1,2})/(\d{4})", "%d/%m/%Y"), # D/M/YYYY, DD/M/YYYY, D/MM/YYYY
            (r"(\d{1,2})-(\d{1,2})-(\d{4})", "%d-%m-%Y"), # D-M-YYYY, DD-M-YYYY, D-MM-YYYY
            (r"(\d{2})\.(\d{2})\.(\d{2})", "%d.%m.%y"), # DD.MM.YY
            (r"(\d{2})/(\d{2})/(\d{2})", "%d/%m/%y"), # DD/MM/YY
            (r"(\d{2})-(\d{2})-(\d{2})", "%d-%m-%y"), # DD-MM-YY
            (r"(\d{1,2})\.(\d{1,2})\.(\d{2})", "%d.%m.%y"), # D.M.YY, DD.M.YY, D.MM.YY
            (r"(\d{1,2})/(\d{1,2})/(\d{2})", "%d/%m/%y"), # D/M/YY, DD/M/YY, D/MM/YY
            (r"(\d{1,2})-(\d{1,2})-(\d{2})", "%d-%m-%y"), # D-M-YY, DD-M-YY, D-MM-YY
            # Add other common formats if needed (e.g., "January 21, 1969")
        ]

        for pattern, date_format in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    # Attempt to parse the date using the specified format
                    date_obj = datetime.strptime(match.group(0), date_format)
                    return date_obj.strftime("%Y-%m-%d")
                except ValueError as e:
                    logger.warning(f"Date format '{date_format}' matched but couldn't parse '{match.group(0)}'. Error: {e}")
                    # Continue searching other patterns
                except Exception as e:
                     logger.error(f"Unexpected error parsing date '{match.group(0)}' with format '{date_format}': {e}")
                     # Continue searching other patterns

        logger.info(f"No date pattern matched in text: '{text[:100]}...'")
        return None # Return None if no pattern matched or parsing failed

    def get_murli_type(self, text):
        """
        Determines if the text indicates an 'Avyakt' Murli.
        Args:
            text (str): The text to check.
        Returns:
            bool: True if 'avyakt' or 'अव्यक्त' is found, False otherwise.
        """
        # Check for both Roman script (case-insensitive) and Devanagari script
        if 'avyakt' in text.lower() or 'अव्यक्त' in text:
            return True
        return False

    def split_text(self, documents, chunk_size=1000, chunk_overlap=200):
        """Splits the documents into chunks using RecursiveCharacterTextSplitter."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(texts)} chunks using RecursiveCharacterTextSplitter")
        return texts

    def semantic_chunking(self, documents, model_name="sentence-transformers/all-MiniLM-L6-v2",
                          chunk_size=128):
        """
        Performs semantic chunking on the input documents using a sentence transformer model.
        Args:
            documents (list): A list of LangChain Document objects.
            model_name (str): The name of the sentence transformer model to use.
            chunk_size (int): The desired maximum size of each chunk in tokens.
        Returns:
            list: A list of LangChain Document objects representing the semantically chunked text.
        """
        logger.info(f"Performing semantic chunking using model: {model_name} with chunk size : {chunk_size} tokens")
        # Initialize the sentence transformer text splitter
        try:
            splitter = SentenceTransformersTokenTextSplitter(model_name=model_name, chunk_overlap=0, tokens_per_chunk=chunk_size)
            # Split the documents into semantically meaningful chunks
            chunks = splitter.split_documents(documents)
            logger.info(f"Split documents into {len(chunks)} chunks using semantic chunking")
            return chunks
        except Exception as e:
            logger.error(f"Error during semantic chunking: {e}")
            # Consider re-raising or returning empty list based on desired behavior
            # raise # Re-raise the exception
            return [] # Return empty list to indicate failure but allow continuation
