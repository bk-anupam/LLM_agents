# /home/bk_anupam/code/LLM_agents/RAG_BOT/vector_store.py
from collections import defaultdict
import os
import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from RAG_BOT.src.logger import logger
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.processing.document_processor import DocumentProcessor
from typing import Optional


class VectorStore:
    def __init__(self, persist_directory=None, config: Optional[Config] = None):
        self.config = config or Config()
        self.persist_directory = persist_directory or self.config.VECTOR_STORE_PATH        
        # Initialize the embedding model once.
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL_NAME)
        logger.info("Embedding model initialized successfully.")
        # Create or load the Chroma vector database.        
        os.makedirs(self.persist_directory, exist_ok=True)
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            try:
                self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
                logger.info(f"Existing vector store loaded successfully from: {self.persist_directory}")
            except Exception as e:
                logger.error(f"Error loading existing vector store from {self.persist_directory}: {e}", exc_info=True)
                # Consider if creating a new one is the right fallback or if it should raise
                logger.warning(f"Attempting to create a new vector store at {self.persist_directory} due to loading error.")
                try:
                    # Attempt to create a new if loading failed (might indicate corruption)
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
            logger.error(f"Error checking ChromaDB for existing date {date_str} and language {language}: {e}. "
                         f"Assuming document does not exist.", exc_info=True)
            # Decide how to handle errors - returning False assumes it doesn't exist, allowing indexing to proceed.
            return False


    def index_document(self, documents, semantic_chunk=False):
        """
        Chunks and indexes a list of documents, checking first if a document with the same date metadata already exists.
        Returns True if indexing occurred, False if skipped or failed.        
        """
        if not documents:
            logger.warning("Attempted to index an empty list of documents. Skipping.")
            return False

        # Extract date from the first document's metadata (assuming consistency for a single document/murli)
        doc_metadata = documents[0].metadata
        extracted_date = doc_metadata.get('date') 
        extracted_language = doc_metadata.get('language') 
        source_file = doc_metadata.get('source', 'N/A') 
        # Determine source for logging
        if source_file and os.path.isabs(source_file) and os.path.exists(source_file):
            log_source = os.path.basename(source_file)
        else:
            log_source = "web_content"
        
        # 1. Check if document already exists using the helper method
        if self.document_exists(extracted_date, extracted_language):
            logger.info(f"Document with date {extracted_date} and language {extracted_language} "
                        f"(source: {log_source}) already indexed. Skipping.")
            return False 

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
                 logger.warning(f"No text chunks generated after processing {log_source}. Nothing to index.")
                 return False 

            self.add_documents(texts)
            logger.info(f"Successfully indexed {len(texts)} chunks from {log_source}.")
            return True 

        except Exception as e:
            logger.error(f"Error during chunking or adding documents for {log_source}: {e}", exc_info=True)
            return False

    
    def log_all_indexed_metadata(self):
        """
        Retrieves and logs a structured, elegant summary of all indexed documents,
        grouped by language, type, and year, including month-day details.
        The entire summary is logged in a single, multi-line statement.
        """
        if not hasattr(self, 'vectordb') or self.vectordb is None:
            logger.error("VectorDB instance not available for metadata retrieval.")
            return
        try:
            all_data = self.vectordb.get(include=['metadatas'])
            
            if not all_data or not all_data.get('ids'):
                logger.info("ChromaDB index appears to be empty. No metadata to retrieve.")
                return

            all_metadatas = all_data.get('metadatas', [])
            total_chunks = len(all_data['ids'])
            
            if not all_metadatas:
                logger.warning("Retrieved document IDs but no corresponding metadata.")
                return

            # --- Data Aggregation ---
            summary_data = defaultdict(lambda: {
                'total_chunks': 0,
                'types': defaultdict(lambda: {
                    'count': 0,
                    'dates': [],
                    'years': defaultdict(lambda: defaultdict(set))
                })
            })

            for metadata in all_metadatas:
                lang = metadata.get('language', 'N/A')
                is_avyakt = metadata.get('is_avyakt')
                date_str = metadata.get('date')

                summary_data[lang]['total_chunks'] += 1
                
                type_key = "Avyakt" if is_avyakt is True else "Sakar/Other" if is_avyakt is False else "Unknown Status"
                
                type_summary = summary_data[lang]['types'][type_key]
                type_summary['count'] += 1

                if date_str:
                    try:
                        dt_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                        type_summary['dates'].append(dt_obj)
                        type_summary['years'][dt_obj.year][dt_obj.month].add(dt_obj.day)
                    except (ValueError, TypeError):
                        logger.debug(f"Could not parse date '{date_str}' for summary.")

            # --- Build the Summary String ---
            summary_lines = [
                "\n--- Indexed Documents Summary ---",
                f"Total Chunks Indexed: {total_chunks}"
            ]

            month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                         7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

            for lang, lang_data in sorted(summary_data.items()):
                summary_lines.append("") # Blank line for spacing
                summary_lines.append(f"Language: {lang} (Total Chunks: {lang_data['total_chunks']})")
                
                for type_key, type_data in sorted(lang_data['types'].items()):
                    summary_lines.append(f"  - Type: {type_key}")
                    summary_lines.append(f"    - Chunk Count: {type_data['count']}")
                    
                    if type_data['dates']:
                        min_date = min(type_data['dates']).strftime('%Y-%m-%d')
                        max_date = max(type_data['dates']).strftime('%Y-%m-%d')
                        summary_lines.append(f"    - Date Range: {min_date} to {max_date}")
                    
                    if type_data['years']:
                        summary_lines.append("    - Documents per Year:")
                        for year, months_data in sorted(type_data['years'].items()):
                            monthly_breakdown = []
                            for month_num, days_set in sorted(months_data.items()):
                                month_name = month_map.get(month_num, '???')
                                days_str = ",".join(map(str, sorted(list(days_set))))
                                monthly_breakdown.append(f"{month_name}({days_str})")
                            
                            summary_lines.append(f"      - {year}: {', '.join(monthly_breakdown)}")

            summary_lines.append("--- End of Summary ---")
            
            # --- Log the entire summary in one go ---
            logger.info("\n".join(summary_lines))

        except Exception as e:
            logger.error(f"Error generating indexed metadata summary: {e}", exc_info=True)
    

    def query_index(self, query, k=25, date_filter=None, language: str = None):
        # Ensure vectordb is initialized
        if not hasattr(self, 'vectordb') or self.vectordb is None:
            logger.error("VectorDB not initialized. Cannot perform query.")
            return "Error: Vector Store is not available."
        
        search_kwargs = {"k": k}
        filters = []

        # Build filter for date
        if date_filter:
            try:
                filter_date = datetime.datetime.strptime(date_filter, '%Y-%m-%d')
                formatted_date = filter_date.strftime('%Y-%m-%d')
                logger.info(f"Applying date filter: {formatted_date}")
                filters.append({"date": formatted_date})
            except ValueError:
                logger.error(f"Invalid date format provided: {date_filter}. Should be YYYY-MM-DD.")
                return "Error: Invalid date format for filter. Please use YYYY-MM-DD."
        
        # Build filter for language
        if language:
            logger.info(f"Applying language filter: {language}")
            filters.append({"language": language})

        # If any filter is set, add to search_kwargs
        if filters:
            if len(filters) > 1:
                # Use $and for multiple filters, which is the correct format for ChromaDB
                search_kwargs["filter"] = {"$and": filters}
            else:
                # Use the single filter directly
                search_kwargs["filter"] = filters[0]

        try:
            retriever = self.vectordb.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
            retrieved_docs = retriever.invoke(query)
            if not retrieved_docs:
                logger.info("No relevant documents found for the query.")
                return "No relevant documents found for the query."

            # --- Group by (date, language) and sort by seq_no ---
            from collections import defaultdict

            murli_groups = defaultdict(list)
            for doc in retrieved_docs:
                date = doc.metadata.get('date', 'N/A')
                lang = doc.metadata.get('language', 'N/A')
                seq_no = doc.metadata.get('seq_no', None)
                murli_groups[(date, lang)].append((seq_no, doc))

            context = ""
            for (date, lang), chunks in murli_groups.items():
                # Sort by seq_no (handle None gracefully)
                sorted_chunks = sorted(chunks, key=lambda x: (x[0] if x[0] is not None else 0))
                # logger.info(f"Document Date: {date}, Language: {lang}")
                context += f"Document Date: {date}, Language: {lang}\n"
                for seq_no, doc in sorted_chunks:
                    # logger.info(f"Chunk {seq_no}:\n{doc.page_content}\n\n")
                    context += f"Chunk {seq_no}:\n{doc.page_content}\n\n"
            logger.info(f"Context for LLM:\n{context}")
            return context if context else "No relevant documents found for the query."
        except Exception as e:
            logger.error(f"Error during query execution: {e}", exc_info=True)
            return "Sorry, an error occurred while processing your query."