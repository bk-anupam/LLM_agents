import re
import os
import sys
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.documents import Document
from RAG_BOT.logger import logger


class DocumentProcessor:
    """
    Base class for processing documents (PDF, HTM, etc.) to extract text,
    metadata, and split content into chunks.
    """

    def _devanagari_to_ascii_digits(self, devanagari_string: str) -> str:
        """Converts Devanagari numerals in a string to ASCII digits."""
        mapping = {
            '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
            '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
        }
        return "".join(mapping.get(char, char) for char in devanagari_string)

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
            (r"([०-९]{4})-([०-९]{2})-([०-९]{2})", "%Y-%m-%d"), # YYYY-MM-DD (Devanagari)

            (r"(\d{2})/(\d{2})/(\d{4})", "%d/%m/%Y"), # DD/MM/YYYY
            (r"([०-९]{2})/([०-९]{2})/([०-९]{4})", "%d/%m/%Y"), # DD/MM/YYYY (Devanagari)

            (r"(\d{2})\.(\d{2})\.(\d{4})", "%d.%m.%Y"), # DD.MM.YYYY
            (r"([०-९]{2})\.([०-९]{2})\.([०-९]{4})", "%d.%m.%Y"), # DD.MM.YYYY (Devanagari)

            (r"(\d{1,2})\.(\d{1,2})\.(\d{4})", "%d.%m.%Y"), # D.M.YYYY, DD.M.YYYY, D.MM.YYYY
            (r"([०-९]{1,2})\.([०-९]{1,2})\.([०-९]{4})", "%d.%m.%Y"), # D.M.YYYY (Devanagari)

            (r"(\d{1,2})/(\d{1,2})/(\d{4})", "%d/%m/%Y"), # D/M/YYYY, DD/M/YYYY, D/MM/YYYY
            (r"([०-९]{1,2})/([०-९]{1,2})/([०-९]{4})", "%d/%m/%Y"), # D/M/YYYY (Devanagari)

            (r"(\d{1,2})-(\d{1,2})-(\d{4})", "%d-%m-%Y"), # D-M-YYYY, DD-M-YYYY, D-MM-YYYY
            (r"([०-९]{1,2})-([०-९]{1,2})-([०-९]{4})", "%d-%m-%Y"), # D-M-YYYY (Devanagari)

            (r"(\d{2})\.(\d{2})\.(\d{2})", "%d.%m.%y"), # DD.MM.YY
            (r"([०-९]{2})\.([०-९]{2})\.([०-९]{2})", "%d.%m.%y"), # DD.MM.YY (Devanagari)

            (r"(\d{2})/(\d{2})/(\d{2})", "%d/%m/%y"), # DD/MM/YY
            (r"([०-९]{2})/([०-९]{2})/([०-९]{2})", "%d/%m/%y"), # DD/MM/YY (Devanagari)

            (r"(\d{2})-(\d{2})-(\d{2})", "%d-%m-%y"), # DD-MM-YY
            (r"([०-९]{2})-([०-९]{2})-([०-९]{2})", "%d-%m-%y"), # DD-MM-YY (Devanagari)

            (r"(\d{1,2})\.(\d{1,2})\.(\d{2})", "%d.%m.%y"), # D.M.YY, DD.M.YY, D.MM.YY
            (r"([०-९]{1,2})\.([०-९]{1,2})\.([०-९]{2})", "%d.%m.%y"), # D.M.YY (Devanagari)

            (r"(\d{1,2})/(\d{1,2})/(\d{2})", "%d/%m/%y"), # D/M/YY, DD/M/YY, D/MM/YY
            (r"([०-९]{1,2})/([०-९]{1,2})/([०-९]{2})", "%d/%m/%y"), # D/M/YY (Devanagari)

            (r"(\d{1,2})-(\d{1,2})-(\d{2})", "%d-%m-%y"), # D-M-YY, DD-M-YY, D-MM-YY
            (r"([०-९]{1,2})-([०-९]{1,2})-([०-९]{2})", "%d-%m-%y"), # D-M-YY (Devanagari)
            # Add other common formats if needed (e.g., "January 21, 1969")
        ]

        for pattern, date_format in date_patterns:
            match = re.search(pattern, text)
            if match:
                matched_date_str = match.group(0)
                ascii_date_str = self._devanagari_to_ascii_digits(matched_date_str)
                try:
                    # Attempt to parse the date using the specified format
                    date_obj = datetime.strptime(ascii_date_str, date_format)
                    return date_obj.strftime("%Y-%m-%d")
                except ValueError as e:
                    logger.warning(f"Date format '{date_format}' matched for '{matched_date_str}' (converted to '{ascii_date_str}'), but couldn't parse. Error: {e}")
                    # Continue searching other patterns
                except Exception as e:
                     logger.error(f"Unexpected error parsing date '{matched_date_str}' (converted to '{ascii_date_str}') with format '{date_format}': {e}")
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
                          chunk_size=1000, chunk_overlap=0):
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
