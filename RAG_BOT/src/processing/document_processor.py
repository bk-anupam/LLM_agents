from typing import List
import RAG_BOT.src.utils as utils
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.documents import Document
from RAG_BOT.src.logger import logger

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

    def split_text(self, documents: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        """
        Splits the combined content of the input documents (assumed to be pages/parts of a single logical Murli)
        into chunks using RecursiveCharacterTextSplitter, adding 'seq_no' metadata.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        full_content = ""
        primary_metadata = {}
        if documents:
            # Use metadata from the first document as representative for the Murli
            primary_metadata = documents[0].metadata.copy()
            primary_metadata.pop('page', None) # Remove page-specific meta if present
            primary_metadata.pop('total_pages', None)

            for doc_page in documents:
                full_content += doc_page.page_content + "\n\n" # Add separator

        if not full_content.strip():
            logger.warning("No content to split after concatenating input documents.")
            return []

        split_content_strings = text_splitter.split_text(full_content)        
        chunked_documents = []
        for i, chunk_text in enumerate(split_content_strings):
            # Each chunk gets a copy of the primary Murli metadata + its sequence number
            chunk_metadata = primary_metadata.copy()
            chunk_metadata['seq_no'] = i + 1
            chunked_documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        logger.info(f"Split combined content into {len(chunked_documents)} chunks with seq_no, using RecursiveCharacterTextSplitter.")
        return chunked_documents


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
        logger.info(f"Performing semantic chunking with seq_no, using model: {model_name} with chunk size: {chunk_size} tokens")                    
        try:
            # chunk_overlap for SentenceTransformersTokenTextSplitter is usually 0.
            splitter = SentenceTransformersTokenTextSplitter(
                model_name=model_name,
                chunk_overlap=0, # Default for this splitter
                tokens_per_chunk=chunk_size
            )
            chunks = splitter.split_documents(documents)

            # Add seq_no to each generated chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata['seq_no'] = i + 1
            logger.info(f"Split content into {len(chunks)} chunks with seq_no, using semantic chunking.")
            return chunks
        except Exception as e:
            logger.error(f"Error during semantic chunking: {e}")
            return [] # Return empty list to indicate failure but allow continuation
