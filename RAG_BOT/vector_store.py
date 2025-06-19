# /home/bk_anupam/code/LLM_agents/RAG_BOT/vector_store.py
from collections import defaultdict
import os
import sys
import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from RAG_BOT.logger import logger
from RAG_BOT.config import Config
from RAG_BOT.document_processor import DocumentProcessor
from typing import Optional


class VectorStore:
    def __init__(self, persist_directory=None, config: Optional[Config] = None):
        self.config = config or Config()
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
        self.document_processor = DocumentProcessor()
        

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


    def document_exists(self, date_str: str, language: str) -> bool:
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


    def index_document(self, documents, semantic_chunk=False):
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
        extracted_date = doc_metadata.get('date') 
        extracted_language = doc_metadata.get('language') 
        source_file = doc_metadata.get('source', 'N/A') # Get source for logging

        # 1. Check if document already exists using the helper method
        if self.document_exists(extracted_date, extracted_language):
            logger.info(f"Document with date {extracted_date} and language {extracted_language} (source: {os.path.basename(source_file)}) already indexed. Skipping.")
            return False # Indicate skip

        # 2. If not existing, proceed with chunking and adding
        logger.info(f"Proceeding with chunking and indexing for {os.path.basename(source_file)}.")
        try:
            if semantic_chunk:
                texts = self.document_processor.semantic_chunking(
                    documents, 
                    chunk_size=self.config.CHUNK_SIZE, 
                    chunk_overlap=self.config.CHUNK_OVERLAP,
                    model_name=self.config.EMBEDDING_MODEL_NAME
                )
            else:
                texts = self.document_processor.split_text(
                    documents, 
                    chunk_size=self.config.CHUNK_SIZE, 
                    chunk_overlap=self.config.CHUNK_OVERLAP
                )
                    
            if not texts:
                 logger.warning(f"No text chunks generated after processing {os.path.basename(source_file)}. Nothing to index.")
                 return False # Indicate skip/failure

            self.add_documents(texts)
            logger.info(f"Successfully indexed {len(texts)} chunks from {os.path.basename(source_file)}.")
            return True # Indicate successful indexing

        except Exception as e:
            logger.error(f"Error during chunking or adding documents for {os.path.basename(source_file)}: {e}", exc_info=True)
            return False # Indicate failure

    # _move_indexed_file method has been removed as its logic is now in FileManager.
    # index_directory method has been removed as its logic is now in DocumentIndexer.

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
                    # Sort by date, then is_avyakt (converted to string for comparison), then language for consistent logging
                    # Use a custom key to handle potential non-boolean types for is_avyakt during sorting
                    for (date, is_avyakt, language), count in sorted(metadata_summary.items(), key=lambda item: (item[0][0], str(item[0][1]), item[0][2])):
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
                    self.config.get_system_prompt(language_code="en") + # Call on instance and provide language_code
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