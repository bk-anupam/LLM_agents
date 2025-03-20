import os
import re
from datetime import datetime
from logger import logger
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.documents import Document

def extract_date_from_text(text):
    """
    Attempts to extract a date from the given text and returns it in YYYY-MM-DD format.
    Args:
        text (str): The text to search for a date.
    Returns:
        str or None: The extracted date in YYYY-MM-DD format if found, otherwise None.
    """
    # Common date patterns (can be expanded)
    date_patterns = [
        r"(\d{4})-(\d{2})-(\d{2})",  # YYYY-MM-DD              
        r"(\d{2})/(\d{2})/(\d{4})", # DD/MM/YYYY    
        r"(\d{2})/(\d{2})/(\d{2})" # DD/MM/YY
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                if pattern == r"(\d{4})-(\d{2})-(\d{2})":
                    year, month, day = match.groups()                
                elif pattern == r"(\d{2})/(\d{2})/(\d{4})":
                    day, month, year = match.groups()  
                elif pattern == r"(\d{2})/(\d{2})/(\d{2})":
                    day, month, year = match.groups()
                    if int(year) < 24:
                        year = "20" + year
                    else:
                        year = "19" + year
                                  
                if isinstance(month, str):
                    if len(month) > 3:
                        month_num = datetime.strptime(month, "%B").month
                    elif len(month) == 3:
                        month_num = datetime.strptime(month, "%b").month
                    else:
                        month_num = int(month)

                date_obj = datetime(int(year), month_num, int(day))
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                logger.error(f"Date format not understood : {match.group(0)}")
                return None  # Return None if date format is not understood
    return None

def load_pdf(pdf_path):
    """Loads a PDF document from the given path and extracts the date metadata."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    documents = []
    for page in pages:
        text = page.page_content        
        date = extract_date_from_text(text[:300])  # Check first 300 characters for date
        metadata = page.metadata
        if date:
            metadata["date"] = date
            logger.info(f"Found date: {date}")
        documents.append(Document(page_content=text, metadata=metadata))
    logger.info(f"Loaded {len(documents)} documents from {pdf_path}")
    return documents

def split_text(documents, chunk_size=1000, chunk_overlap=200):
    """Splits the documents into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(texts)} chunks")
    return texts

def create_embeddings():
    """Creates embeddings using HuggingFaceEmbeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Or another suitable model
    return embeddings

def semantic_chunking(documents, model_name="sentence-transformers/all-MiniLM-L6-v2", 
                      chunk_size=128):
    """
    Performs semantic chunking on the input documents using a sentence transformer model.
    Args:
        documents (list): A list of LangChain Document objects.
        model_name (str): The name of the sentence transformer model to use.
        chunk_size (int): The desired maximum size of each chunk.
    Returns:
        list: A list of LangChain Document objects representing the semantically chunked text.
    """
    logger.info(f"Performing semantic chunking using model: {model_name} with chunk size : {chunk_size}")    
    # Initialize the sentence transformer text splitter
    try:
        splitter = SentenceTransformersTokenTextSplitter(model_name=model_name, chunk_overlap=0, tokens_per_chunk=chunk_size)    
        # Split the documents into semantically meaningful chunks
        chunks = splitter.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks using semantic chunking")
        return chunks
    except Exception as e:
        logger.error(f"Error during semantic chunking: {e}")
        raise

if __name__ == "__main__":
    pdf_path = "/home/bk_anupam/code/ML/NLP/LLMs/RAG/RAG_TELEGRAM_BOT/uploads/07_AV-E-06.02_1969.pdf"
    documents = load_pdf(pdf_path)
    chunks = semantic_chunking(documents)