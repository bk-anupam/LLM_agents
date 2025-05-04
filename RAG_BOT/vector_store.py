# /home/bk_anupam/code/LLM_agents/RAG_BOT/vector_store.py
from collections import defaultdict
import os
import sys
import datetime
import shutil
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from RAG_BOT.logger import logger
from RAG_BOT.config import Config
from RAG_BOT.pdf_processor import PdfProcessor
from RAG_BOT.htm_processor import HtmProcessor


class VectorStore:
    def __init__(self, persist_directory=None):
        self.config = Config()
        self.persist_directory = persist_directory or self.config.VECTOR_STORE_PATH        
        # Initialize the embedding model once.
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL_NAME)
        logger.info("Embedding model initialized successfully.")
        # Create or load the Chroma vector database.
        # Added more robust error handling and directory creation
        os.makedirs(self.persist_directory, exist_ok=True) # Ensure directory exists before checking content
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            try:
                self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
                logger.info(f"Existing vector store loaded successfully from: {self.persist_directory}")
            except Exception as e:
                logger.error(f"Error loading existing vector store from {self.persist_directory}: {e}", exc_info=True)
                # Consider if creating a new one is the right fallback or if it should raise
                logger.warning(f"Attempting to create a new vector store at {self.persist_directory} due to loading error.")
                try:
                    # Attempt to create anew if loading failed (might indicate corruption)
                    self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
                    logger.info("New vector store created after load failure.")
                except Exception as inner_e:
                    logger.critical(f"Failed to create new vector store after load failure: {inner_e}", exc_info=True)
                    raise inner_e # Re-raise critical failure
        else:
            try:
                self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
                logger.info(f"New vector store created successfully at: {self.persist_directory}")
            except Exception as e:
                 logger.critical(f"Failed to create new vector store at {self.persist_directory}: {e}", exc_info=True)
                 raise e # Re-raise critical failure
        # Initialize document processors
        self.pdf_processor = PdfProcessor()
        self.htm_processor = HtmProcessor()

    def get_vectordb(self):
        return self.vectordb

    def add_documents(self, texts):
        if not texts:
            logger.warning("Attempted to add an empty list of documents. Skipping.")
            return
        source = texts[0].metadata.get('source', 'N/A') # Get source from first doc for logging
        try:
            self.vectordb.add_documents(documents=texts)
            logger.info(f"Vector store updated with {len(texts)} document chunks from source: {os.path.basename(source)}") # Log only basename
        except Exception as e:
            logger.error(f"Failed to add documents from source {os.path.basename(source)} to ChromaDB: {e}", exc_info=True)


    def _document_exists(self, date_str: str, language: str) -> bool:
        """
        Checks if any document with the given date string and language already exists in the index.
        """
        if not date_str:
            # If date extraction failed, we assume it doesn't exist to allow indexing.
            # The alternative is to skip indexing files without dates.
            logger.warning("Cannot check for existing document without a date string. Assuming it does not exist.")
            return False
        if not language:
            # Similarly, if language is missing, we cannot perform the combined check.
            logger.warning("Cannot check for existing document without language metadata. Assuming it does not exist.")
            return False

        logger.debug(f"Checking for existing documents with date: {date_str} and language: {language}")
        try:
            # Check if vectordb is initialized
            if not hasattr(self, 'vectordb') or self.vectordb is None:
                logger.error("VectorDB not initialized. Cannot check for existing documents.")
                return False # Cannot check, assume not found

            existing_docs = self.vectordb.get(
                # Use $and operator for multiple metadata filters
                where={
                    "$and": [
                        {"date": date_str},
                        {"language": language}
                    ]                },
                limit=1, # We only need to know if at least one exists
                include=[] # We don't need metadata or documents, just the count implicitly
            )
            # Check if the 'ids' list is not empty
            if existing_docs and existing_docs.get('ids'):
                logger.debug(f"Document with date {date_str} and language {language} found in the index.")
                return True
            else:
                logger.debug(f"No document found with date {date_str} and language {language}.")
                return False
        except Exception as e:
            logger.error(f"Error checking ChromaDB for existing date {date_str} and language {language}: {e}. Assuming document does not exist.", exc_info=True)
            # Decide how to handle errors - returning False assumes it doesn't exist, allowing indexing to proceed.
            return False


    def _index_document(self, documents, chunk_size=1000, chunk_overlap=200, semantic_chunk=True):
        """
        Chunks and indexes a list of documents, checking first if a document with the same date metadata already exists.
        Returns True if indexing occurred, False if skipped or failed.
        This is a private helper method.
        """
        if not documents:
            logger.warning("Attempted to index an empty list of documents. Skipping.")
            return False

        # Extract date from the first document's metadata (assuming consistency for a single document/murli)
        doc_metadata = documents[0].metadata
        extracted_date = doc_metadata.get('date') # Get the date string
        extracted_language = doc_metadata.get('language') # Get the language string
        source_file = doc_metadata.get('source', 'N/A') # Get source for logging

        # 1. Check if document already exists using the helper method
        if self._document_exists(extracted_date, extracted_language):
            logger.info(f"Document with date {extracted_date} and language {extracted_language} (source: {os.path.basename(source_file)}) already indexed. Skipping.")
            return False # Indicate skip

        # 2. If not existing, proceed with chunking and adding
        logger.info(f"Proceeding with chunking and indexing for {os.path.basename(source_file)}.")
        try:
            if semantic_chunk:
                texts = (
                    self.pdf_processor.semantic_chunking(documents)
                    if documents and documents[0].metadata.get('source', '').lower().endswith('.pdf')
                    else self.htm_processor.semantic_chunking(documents)
                )
            else:
                texts = (
                    self.pdf_processor.split_text(documents, chunk_size, chunk_overlap)
                    if documents and documents[0].metadata.get('source', '').lower().endswith('.pdf')
                    else self.htm_processor.split_text(documents, chunk_size, chunk_overlap)
                )

            if not texts:
                 logger.warning(f"No text chunks generated after processing {os.path.basename(source_file)}. Nothing to index.")
                 return False # Indicate skip/failure

            self.add_documents(texts)
            return True # Indicate successful indexing

        except Exception as e:
            logger.error(f"Error during chunking or adding documents for {os.path.basename(source_file)}: {e}", exc_info=True)
            return False # Indicate failure

    def _move_indexed_file(self, source_path: str, language: str):
        """
        Moves a successfully indexed file to the 'indexed_data' directory,
        preserving the language subdirectory structure.
        """
        if self.config.INDEXED_DATA_PATH is None:
            logger.error("INDEXED_DATA_PATH not configured. Cannot move indexed files.")
            return False

        if not source_path or not os.path.exists(source_path):
            logger.warning(f"Source file path invalid or file does not exist: {source_path}. Cannot move.")
            return False

        try:
            file_name = os.path.basename(source_path)
            destination_subdir = os.path.join(self.config.INDEXED_DATA_PATH, language)
            destination_path = os.path.join(destination_subdir, file_name)

            # Ensure the destination subdirectory exists
            os.makedirs(destination_subdir, exist_ok=True)

            shutil.move(source_path, destination_path)
            logger.info(f"Successfully moved indexed file '{file_name}' to: {destination_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to move file {source_path} to {destination_path}: {e}", exc_info=True)
            return False


    def index_directory(self, base_data_path: str = None):
        """
        Recursively finds PDF files in 'english' subdirectory and HTM files in 'hindi'
        subdirectory of the base_data_path, adds language metadata, and indexes them,
        skipping those already indexed based on date metadata.
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

                # Add language metadata if documents were loaded
                if documents:
                    for doc in documents:
                        doc.metadata['language'] = language
                    logger.debug(f"Added language metadata '{language}' to document: {os.path.basename(file_path)}")
                else:
                    logger.warning(f"No documents loaded for file: {file_path}. Skipping indexing for this file.")
                    continue # Skip to next file if no documents loaded

                # _index_document now handles the rest of the process (checking date, chunking, adding)
                was_indexed = self._index_document(
                    documents, # Pass the list of documents (potentially just one for HTM)
                    semantic_chunk=self.config.SEMANTIC_CHUNKING
                    # Pass other params like chunk_size if needed from config
                )
                if was_indexed:
                    indexed_count += 1
                    # Move the file if indexing was successful
                    if self._move_indexed_file(file_path, language):
                        moved_count += 1
                else:
                    # This counts skips due to already existing docs or failures within _index_document
                    skipped_count += 1
            except Exception as e:
                # Catch unexpected errors directly during the loop iteration
                logger.error(f"Unhandled exception during indexing attempt for {file_path}: {e}", exc_info=True)
                failed_count += 1

        logger.info(f"Directory indexing complete for: {base_data_path}")
        logger.info(f"Summary: Indexed={indexed_count}, Moved={moved_count}, Skipped/IndexFailed={skipped_count}, UnhandledFailures={failed_count}")


    def log_all_indexed_metadata(self):
        """
        Retrieves and logs the date, is_avyakt, and language metadata for ALL indexed documents.
        Groups and counts by (date, is_avyakt, language).
        """
        if not hasattr(self, 'vectordb') or self.vectordb is None:
            logger.error("VectorDB instance not available for metadata retrieval.")
            return
        try:
            logger.info("Attempting to retrieve metadata for ALL indexed documents...")
            all_data = self.vectordb.get(include=['metadatas'])
            if all_data and all_data.get('ids'):
                all_metadatas = all_data.get('metadatas', [])
                total_docs = len(all_data['ids'])
                logger.info(f"Retrieved metadata for {total_docs} documents.")
                if not all_metadatas:
                    logger.warning("Retrieved document IDs but no corresponding metadata.")
                    return
                # Structure: {(date, is_avyakt, language): count}
                metadata_summary = defaultdict(int)
                missing_metadata_count = 0
                for metadata in all_metadatas:
                    date = metadata.get('date', 'N/A')
                    is_avyakt = metadata.get('is_avyakt', 'N/A')
                    language = metadata.get('language', 'N/A')
                    if date == 'N/A' and is_avyakt == 'N/A' and language == 'N/A':
                        missing_metadata_count += 1
                    else:
                        metadata_summary[(date, is_avyakt, language)] += 1
                logger.info("--- Logging Date, Avyakt Status, and Language for All Indexed Documents ---")
                if metadata_summary:
                    # Sort by date, then is_avyakt, then language for consistent logging
                    for (date, is_avyakt, language), count in sorted(metadata_summary.items()):
                        avyakt_str = "Avyakt" if is_avyakt is True else "Sakar/Other" if is_avyakt is False else "Unknown Status"
                        logger.info(f"Date: {date} - Type: {avyakt_str}, Language: {language}, Count: {count}")
                else:
                    logger.info("No documents with date, is_avyakt, or language metadata found.")

                if missing_metadata_count > 0:
                    logger.warning(f"Found {missing_metadata_count} documents missing date, is_avyakt, and language metadata.")

                logger.info("--- Finished Logging All Indexed Metadata ---")
            else:
                logger.info("ChromaDB index appears to be empty. No metadata to retrieve.")
        except Exception as e:
            logger.error(f"Error retrieving all metadata from ChromaDB: {e}", exc_info=True)
    

    def query_index(self, query, chain_type="stuff", k=25, model_name="gemini-2.0-flash", date_filter=None):
        # Ensure vectordb is initialized
        if not hasattr(self, 'vectordb') or self.vectordb is None:
            logger.error("VectorDB not initialized. Cannot perform query.")
            return "Error: Vector Store is not available."

        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
        search_kwargs = {"k": k}

        if date_filter:
            try:
                filter_date = datetime.datetime.strptime(date_filter, '%Y-%m-%d')
                formatted_date = filter_date.strftime('%Y-%m-%d')
                logger.info(f"Applying date filter: {formatted_date}")
                # Use Chroma's metadata filtering syntax
                search_kwargs["filter"] = {"date": formatted_date}
            except ValueError:
                logger.error(f"Invalid date format provided: {date_filter}. Should be YYYY-MM-DD.")
                # Return an error or raise? Returning error message for now.
                return "Error: Invalid date format for filter. Please use YYYY-MM-DD."

        try:
            retriever = self.vectordb.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
            retrieved_docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: '{query[:50]}...'") # Log snippet
            logger.debug(f"Context for LLM: {context}") # Log snippet

            custom_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    Config.get_system_prompt() +
                    "\n\nContext:\n{context}\n\nQuestion: {question}" # Added newlines for clarity
                ),
            )
            chain = custom_prompt | llm | RunnablePassthrough() # Removed RunnablePassthrough, not needed here
            response = chain.invoke({"context": context, "question": query})

            if isinstance(response, AIMessage):
                return response.content
            # Handle potential string or other types returned by the chain
            elif isinstance(response, str):
                 return response
            elif hasattr(response, 'content'): # Check for Langchain Core message types
                 return response.content
            else:
                 logger.warning(f"Unexpected response type from LLM chain: {type(response)}")
                 return str(response) # Fallback to string representation

        except Exception as e:
            logger.error(f"Error during query execution: {e}", exc_info=True)
            return "Sorry, an error occurred while processing your query."    
        

# --- Standalone script functions ---
# Note: These might become less necessary if indexing happens on startup,
# but can be kept for manual re-indexing or testing.

def test_query_index():
    """
    Test querying the index.
    """
    vs = VectorStore()
    # query = "1992-09-24 की मुरली का सार क्या है? कृपया पुरुषार्थ के दृष्टिकोण से हिंदी भाषा में बताएं|"
    query = "दूसरों की चेकिंग करने के बारे में बाबा ने मुरली में क्या बताया है?"
    # Example date - adjust if needed
    test_date = "1992-09-24" # Make sure this date exists in your indexed data for a good test
    logger.info(f"Testing query with date filter: {test_date}")
    try:
        result = vs.query_index(query, k=10, date_filter=test_date)
        print("\n--- Query Result ---")
        print(result)
        print("--- End Query Result ---\n")
    except ValueError as e:
        logger.error(f"Query failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during query test: {e}", exc_info=True)


def index_data():
    """
    Build the index for all PDFs in 'english' and HTM files in 'hindi' subdirectories
    of the configured DATA_PATH.
    """
    config = Config()
    data_dir = config.DATA_PATH # Get base data directory from config
    if not data_dir:
        logger.error("DATA_PATH is not set in the configuration. Cannot start indexing.")
        return

    logger.info(f"Starting MANUAL indexing process for base directory: {data_dir}")

    vs = VectorStore() # Initialize VectorStore
    vs.index_directory(data_dir) # Call index_directory with the base path

    logger.info("Manual indexing process complete.")
    # Log the final state of the index
    # vs.log_all_indexed_metadata()


if __name__ == "__main__":
    # Decide whether to index or test - less relevant if indexing is on startup
    # index_data()
    test_query_index()
