import RAG_BOT.utils as utils
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.documents import Document
from RAG_BOT.logger import logger

class DocumentProcessor:
    """
    Base class for processing documents (PDF, HTM, etc.) to extract text,
    metadata, and split content into chunks.
    """        

    def extract_date_from_text(self, text):
        return utils.extract_date_from_text(text)

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
