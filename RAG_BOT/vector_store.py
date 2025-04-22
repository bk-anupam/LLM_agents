# /home/bk_anupam/code/LLM_agents/RAG_BOT/vector_store.py
from collections import defaultdict
import os
import datetime
import re
from RAG_BOT.logger import logger
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from RAG_BOT.config import Config
from RAG_BOT.pdf_processor import load_pdf, split_text, semantic_chunking
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough


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


    def _document_exists_by_date(self, date_str: str) -> bool:
        """
        Checks if any document with the given date string already exists in the index.
        """
        if not date_str:
            # If date extraction failed, we assume it doesn't exist to allow indexing.
            # The alternative is to skip indexing files without dates.
            logger.warning("Cannot check for existing document without a date string. Assuming it does not exist.")
            return False

        logger.debug(f"Checking for existing documents with date: {date_str}")
        try:
            # Check if vectordb is initialized
            if not hasattr(self, 'vectordb') or self.vectordb is None:
                logger.error("VectorDB not initialized. Cannot check for existing documents.")
                return False # Cannot check, assume not found

            existing_docs = self.vectordb.get(
                where={"date": date_str},
                limit=1, # We only need to know if at least one exists
                include=[] # We don't need metadata or documents, just the count implicitly
            )
            # Check if the 'ids' list is not empty
            if existing_docs and existing_docs.get('ids'):
                logger.debug(f"Document with date {date_str} found in the index.")
                return True
            else:
                logger.debug(f"No document found with date {date_str}.")
                return False
        except Exception as e:
            logger.error(f"Error checking ChromaDB for existing date {date_str}: {e}. Assuming document does not exist.", exc_info=True)
            # Decide how to handle errors - returning False assumes it doesn't exist, allowing indexing to proceed.
            return False


    def build_index(self, pdf_path, chunk_size=1000, chunk_overlap=200, semantic_chunk=True):
        """
        Loads, chunks, and indexes a PDF, checking first if a document with the same date metadata already exists.
        Returns True if indexing occurred, False if skipped or failed.
        """
        logger.info(f"Processing PDF for indexing: {pdf_path}")
        source_file = os.path.basename(pdf_path) # Get filename for logging

        # 1. Load document and extract metadata
        try:
            # Ensure pdf_processor handles potential errors during loading/parsing
            documents = load_pdf(pdf_path)
            if not documents:
                logger.warning(f"No documents extracted from PDF: {source_file}. Skipping.")
                return False # Indicate skip/failure

            # Extract date from the first document's metadata (assuming consistency)
            doc_metadata = documents[0].metadata
            extracted_date = doc_metadata.get('date') # Get the date string

        except FileNotFoundError:
            logger.error(f"PDF file not found: {pdf_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load or extract metadata from PDF: {source_file}. Error: {e}", exc_info=True)
            return False # Indicate failure

        # 2. Check if document already exists using the helper method
        if self._document_exists_by_date(extracted_date):
            logger.info(f"Document with date {extracted_date} (source: {source_file}) already indexed. Skipping.")
            return False # Indicate skip

        # 3. If not existing, proceed with chunking and adding
        logger.info(f"Proceeding with chunking and indexing for {source_file}.")
        try:
            if semantic_chunk:
                texts = semantic_chunking(documents)
            else:
                texts = split_text(documents, chunk_size, chunk_overlap)

            if not texts:
                 logger.warning(f"No text chunks generated after processing {source_file}. Nothing to index.")
                 return False # Indicate skip/failure

            self.add_documents(texts)
            return True # Indicate successful indexing

        except Exception as e:
            logger.error(f"Error during chunking or adding documents for {source_file}: {e}", exc_info=True)
            return False # Indicate failure


    # --- New Method for Directory Indexing ---
    def index_directory(self, directory_path: str):
        """
        Recursively finds all PDF files in the given directory and indexes them,
        skipping those already indexed based on date metadata.
        """
        logger.info(f"Starting recursive indexing for directory: {directory_path}")
        if not os.path.isdir(directory_path):
            logger.error(f"Provided path is not a valid directory: {directory_path}")
            return

        pdf_files_found = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files_found.append(os.path.join(root, file))

        if not pdf_files_found:
            logger.warning(f"No PDF files found in directory: {directory_path}")
            return

        logger.info(f"Found {len(pdf_files_found)} PDF files to potentially index.")
        indexed_count = 0
        skipped_count = 0
        failed_count = 0

        for pdf_file in pdf_files_found:
            try:
                # build_index now returns True if indexed, False otherwise
                was_indexed = self.build_index(
                    pdf_file,
                    semantic_chunk=self.config.SEMANTIC_CHUNKING
                    # Pass other params like chunk_size if needed from config
                )
                if was_indexed:
                    indexed_count += 1
                else:
                    # This counts both skips and failures within build_index
                    skipped_count += 1
            except Exception as e:
                # Catch unexpected errors directly during the loop iteration
                logger.error(f"Unhandled exception during indexing attempt for {pdf_file}: {e}", exc_info=True)
                failed_count += 1

        logger.info(f"Directory indexing complete for: {directory_path}")
        logger.info(f"Summary: Indexed={indexed_count}, Skipped/Failed(in build_index)={skipped_count}, Failed(Unhandled)={failed_count}")
    # --- End New Method ---


    def log_all_indexed_metadata(self):
        """Retrieves and logs the date and is_avyakt metadata for ALL indexed documents."""
        if not hasattr(self, 'vectordb') or self.vectordb is None:
            logger.error("VectorDB instance not available for metadata retrieval.")
            return
        try:
            logger.info("Attempting to retrieve metadata for ALL indexed documents...")
            # Use get() without limit to fetch all, include only 'metadatas'
            all_data = self.vectordb.get(include=['metadatas']) # No limit specified
            if all_data and all_data.get('ids'):
                all_metadatas = all_data.get('metadatas', [])
                total_docs = len(all_data['ids'])
                logger.info(f"Retrieved metadata for {total_docs} documents.")
                if not all_metadatas:
                    logger.warning("Retrieved document IDs but no corresponding metadata.")
                    return
                # Use defaultdict for easier counting grouped by date and then by avyakt status
                # Structure: {date: {is_avyakt_status: count}}
                metadata_summary = defaultdict(lambda: defaultdict(int))
                missing_metadata_count = 0 # Count docs missing both date and avyakt
                for metadata in all_metadatas:
                    date = metadata.get('date', 'N/A')
                    is_avyakt = metadata.get('is_avyakt', 'N/A') # Default to 'N/A' if missing

                    if date == 'N/A' and is_avyakt == 'N/A':
                        missing_metadata_count += 1
                    else:
                        # Group by date, then by avyakt status, and count occurrences
                        metadata_summary[date][is_avyakt] += 1
                logger.info("--- Logging Date and Avyakt Status for All Indexed Documents ---")
                if metadata_summary:
                    # Sort by date for chronological logging
                    for date, avyakt_counts in sorted(metadata_summary.items()):
                        logger.info(f"  - Date: {date}")
                        # Sort avyakt statuses (True/False/N/A) for consistent order
                        for is_avyakt, count in sorted(avyakt_counts.items()):
                             # Make the boolean more readable
                            avyakt_str = "Avyakt" if is_avyakt is True else "Sakar/Other" if is_avyakt is False else "Unknown Status"
                            logger.info(f"    - Type: {avyakt_str} (is_avyakt={is_avyakt}), Count: {count}")
                else:
                    logger.info("No documents with date or is_avyakt metadata found.")

                if missing_metadata_count > 0:
                    logger.warning(f"Found {missing_metadata_count} documents missing both date and is_avyakt metadata.")

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

            custom_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    self.config.SYSTEM_PROMPT +
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


    @staticmethod
    def query_llm(query, model_name="gemini-2.0-flash"):
        """Queries the LLM directly and returns the answer."""
        try:
            # Initialize the LLM
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
            # Create a custom prompt using the system prompt text
            # Need to instantiate Config or make SYSTEM_PROMPT static/class variable
            config = Config()
            custom_prompt = PromptTemplate(
                input_variables=["question"],
                template=(
                    config.SYSTEM_PROMPT +
                    "\n\nQuestion: {question}" # Added newline
                ),
            )
            # Format the query using the custom prompt
            formatted_query = custom_prompt.format(question=query)
            # Format the query properly as a message
            messages = [HumanMessage(content=formatted_query)]
            # Get the response
            response = llm.invoke(messages)
            # Extract the content from the response
            return response.content
        except Exception as e:
            logger.error(f"Error querying LLM directly: {str(e)}", exc_info=True)
            return f"Sorry, I encountered an error communicating with the language model: {str(e)}"


# --- Standalone script functions ---
# Note: These might become less necessary if indexing happens on startup,
# but can be kept for manual re-indexing or testing.

def index_data():
    """
    Build the index for all PDFs in a given directory.
    DEPRECATED if using index_directory on startup. Kept for potential manual use.
    """
    # Example directory - should ideally come from config or argument
    pdf_dir = "/home/bk_anupam/code/LLM_agents/RAG_BOT/data/pdfs_to_index" # Example path
    config = Config()
    logger.info(f"Starting MANUAL indexing process for directory: {pdf_dir}")

    vs = VectorStore() # Initialize VectorStore
    vs.index_directory(pdf_dir) # Use the new method

    logger.info("Manual indexing process complete.")
    # Log the final state of the index
    vs.log_all_indexed_metadata()


def test_query_index():
    """
    Test querying the index.
    """
    vs = VectorStore()
    query = "What are the main points regarding remembrance that Baba talks about? Summarize in 2-3 sentences."
    # Example date - adjust if needed
    test_date = "1970-01-18" # Make sure this date exists in your indexed data for a good test
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


# if __name__ == "__main__":
    # Decide whether to index or test - less relevant if indexing is on startup
    # index_data()
    # test_query_index()
