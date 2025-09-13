from langchain_core.tools import tool
from langchain_chroma import Chroma
from typing import Optional, Dict, Any, Callable, List, Tuple 
from RAG_BOT.src.logger import logger
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.context_retrieval.filter_processor import FilterProcessor
from RAG_BOT.src.context_retrieval.bm25_processor import BM25Processor
from RAG_BOT.src.context_retrieval.result_processor import ResultProcessor


class RetrieverExecutor:
    """Handles retrieval operations with fallback mechanisms."""
    
    @staticmethod
    def execute_with_fallback(
        vectordb: Chroma,
        query: str,
        search_args: Dict[str, Any],
        k_initial: int,
        k_fallback: int,
        search_type: str,
        operation_name: str
    ) -> List[Any]:
        """Execute retrieval with fallback for HNSW errors."""
        attempts = [
            (k_initial, f"{operation_name} with k={k_initial}"),
            (k_fallback, f"{operation_name} fallback with k={k_fallback}")
        ]
        
        for k_value, description in attempts:
            if k_value == k_initial or (k_initial != 1 or k_fallback != 1):  # Avoid infinite loop
                try:
                    current_args = {**search_args, "k": k_value}
                    logger.debug(f"{description}: Using search_kwargs: {current_args}")
                    
                    retriever = vectordb.as_retriever(search_type=search_type, search_kwargs=current_args)
                    docs = retriever.invoke(query)
                    
                    # Add retrieval_type to metadata for downstream processing to distinguish
                    # how the document was retrieved (e.g., 'semantic' vs 'bm25').
                    for doc in docs:
                        doc.metadata['retrieval_type'] = 'semantic'

                    logger.info(f"{description} retrieved {len(docs)} docs.")
                    return docs
                    
                except RuntimeError as e:
                    if "Cannot return the results in a contigious 2D array" in str(e):
                        logger.warning(f"HNSW RuntimeError for {description}: {e}")
                        if k_value == k_initial:  # Only continue to fallback on first attempt
                            continue
                    logger.error(f"Unhandled RuntimeError during {description}: {e}", exc_info=True)
                    break
                except Exception as e:
                    logger.error(f"Error during {description}: {e}", exc_info=True)
                    break
        
        return []


def create_context_retriever_tool(vectordb: Chroma, config: Config) -> Callable:
    """Factory function that creates the murli retriever tool with enclosed dependencies."""
    
    bm25_processor = BM25Processor(vectordb)
    result_processor = ResultProcessor()
    
    
    def _handle_date_specific_query(
        active_date: str,
        language: str,
    ) -> Tuple[str, List[Tuple[str, Dict[str, Any]]]]:
        """Handles retrieval for queries with a specific date by reconstructing the full document."""
        logger.info(f"Date filter '{active_date}' provided. Bypassing search and directly reconstructing full Murli.")
        dummy_metadata = [{"date": active_date, "language": language}]
        reconstructed_context = result_processor.reconstruct_murlis(
            dummy_metadata, vectordb, config
        )
        if not reconstructed_context:
            status_msg = f"A Murli for date {active_date} and language {language} was not found in the database."
            logger.warning(status_msg)
            return status_msg, []
        
        status_msg = f"Successfully reconstructed full Murli for date {active_date}."
        logger.info(status_msg)
        return status_msg, reconstructed_context


    def _handle_topical_query(
            query: str, 
            search_kwargs: Dict[str, Any]
    ) -> Tuple[str, List[Tuple[str, Dict[str, Any]]]]:
        """Handles retrieval for topical queries using hybrid search and context reconstruction."""
        logger.info("No date filter provided. Proceeding with hybrid search for topical query.")
        
        # Early validation
        if not query or not query.strip():
            return "Empty query provided.", []
        
        # Process query once upfront
        semantic_query = query.strip()
        normalized_query = semantic_query.lower()
        
        # Configuration 
        k_semantic = config.INITIAL_RETRIEVAL_K
        k_bm25 = config.BM25_TOP_K
        max_corpus_for_bm25 = config.BM25_MAX_CORPUS_SIZE
        search_type = config.SEARCH_TYPE
        k_fallback = getattr(config, "K_FALLBACK", max(1, min(5, k_semantic // 2 if k_semantic > 1 else 1)))
        
        try:
            # 1. Semantic Search
            semantic_docs = RetrieverExecutor.execute_with_fallback(
                vectordb, semantic_query, search_kwargs, k_semantic, k_fallback, search_type, "semantic search"
            )
            logger.info(f"Semantic search retrieved {len(semantic_docs)} chunks.")
            
            # 2. BM25 Search - streamlined logic
            bm25_results = []
            search_filter = search_kwargs.get("filter")
            if search_filter:
                corpus_items = bm25_processor.get_scoped_corpus(search_filter, max_corpus_for_bm25)
                if corpus_items:
                    bm25_results = BM25Processor.search(normalized_query, corpus_items, k_bm25)
                    logger.info(f"BM25 search retrieved {len(bm25_results)} chunks.")
                else:
                    logger.info("No corpus items found for BM25 search.")
            else:
                logger.info("No filter provided for BM25 search, skipping.")
            
            # 3. Combine results with early return
            combined_chunks = result_processor.combine_and_deduplicate(semantic_docs, bm25_results)
            if not combined_chunks:
                return "No relevant chunks found from hybrid retrieval.", []
            
            # 4. Context reconstruction - cleaner logic
            final_results = combined_chunks  # Default to combined chunks
            reconstruction_used = False
            
            # Check if any reconstruction is enabled
            if config.SENTENCE_WINDOW_RECONSTRUCTION or config.RECONSTRUCT_MURLIS:
                chunk_metadatas = [meta for _, meta in combined_chunks]
                reconstructed_context = []
                
                if config.SENTENCE_WINDOW_RECONSTRUCTION:
                    logger.info("Using sentence window context reconstruction.")
                    reconstructed_context = result_processor.reconstruct_from_sentence_windows(
                        chunk_metadatas, vectordb, config
                    )
                elif config.RECONSTRUCT_MURLIS:
                    logger.info("Using full Murli reconstruction.")
                    reconstructed_context = result_processor.reconstruct_murlis(
                        chunk_metadatas, vectordb, config
                    )
                
                # Use reconstructed context if successful
                if reconstructed_context:
                    final_results = reconstructed_context
                    reconstruction_used = True
                else:
                    logger.warning("Context reconstruction failed, using combined chunks.")
            
            # 5. Build status message - unified logic
            base_message = f"Retrieved {len(semantic_docs)} semantic + {len(bm25_results)} BM25 chunks."
            if reconstruction_used:
                status_msg = f"{base_message} Reconstructed context for {len(final_results)} retrieved chunks."
            else:
                status_msg = f"{base_message} Returning {len(final_results)} unique chunks (no reconstruction)."
            
            logger.info(status_msg)
            return status_msg, final_results
            
        except Exception as e:
            logger.error(f"Error during topical query processing: {e}", exc_info=True)
            return f"Search failed: {str(e)}", []



    @tool(response_format="content_and_artifact")
    def retrieve_context(
        query: str,
        date_filter: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Tuple[str, List[Tuple[str, Dict[str, Any]]]]:
        """
        Retrieves relevant context snippets from indexed Brahmakumaris murlis using a hybrid approach
        (semantic search + BM25 lexical search if applicable), optionally filtering by date and language.
        Use this tool when the user asks for summaries, details, specific points, or content related to
        a particular date (YYYY-MM-DD), topic, or concept mentioned within the murlis.

        Args:            
            query: The semantic query derived from the user's original question. This should capture the full intent of the user's 
                request and MUST include any date mentioned in the original question as part of the query string itself, in addition 
                to the date being extracted for the 'date_filter' parameter. For example, if the user asks 
                'What is the summary of the murli from 1969-01-18?', the query should be 'summary of the murli from 1969-01-18'.
            date_filter: An optional date string in 'YYYY-MM-DD' format extracted from the user's query to filter documents by date. 
                Provide ONLY if a specific date is mentioned.
            language: The language code ('en' for English, 'hi' for Hindi) inferred from the user's original query. 
                ALWAYS try to infer and provide this parameter based on the user's input language.

        Returns:            
            A tuple containing:
            1. A status string indicating the outcome.
            2. A list of (murli_content, murli_metadata) tuples.
            Returns ("Error retrieving context.", []) if an error occurs.
        """
        logger.info(f"Executing Murli retrieval for query: '{query}', date: {date_filter}, lang: {language}")
        
        # Prepare filters
        search_kwargs, active_date = FilterProcessor.prepare_filters(date_filter, language)        

        # --- Dispatch to the appropriate handler ---
        if active_date and language:
            return _handle_date_specific_query(active_date, language)
        else:
            return _handle_topical_query(query, search_kwargs)
    
    return retrieve_context


# Example usage
if __name__ == '__main__':
    from langchain_huggingface import HuggingFaceEmbeddings
    from RAG_BOT.src.config.config import Config
    
    persist_directory = '/home/bk_anupam/code/LLM_agents/RAG_BOT/chroma_db'
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
    
    try:
        vectordb_instance = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        config_instance = Config()
        
        murli_tool = create_context_retriever_tool(vectordb_instance, config_instance)
        logger.info("Context retriever tool created successfully.")
        
        # --- Test 1: Date-specific query (should bypass search and reconstruct full Murli) ---
        print("\n--- Testing Date-Specific Retrieval ---")
        # Use a date known to be in the test data from integration tests
        date_specific_query = "This query is ignored when a date is provided." # Query text is ignored for date-specific retrieval
        # Ensure the language is also provided, as it's part of the condition
        date_specific_input = {"query": date_specific_query, "date_filter": "2025-09-10", "language": "hi"}
        
        # Call the underlying function directly to get the full tuple output
        status_msg_date, reconstructed_context_date = murli_tool.func(
            query=date_specific_input["query"],
            date_filter=date_specific_input["date_filter"],
            language=date_specific_input["language"]
        )
        
        print(f"Query: '{date_specific_query}'")
        print(f"Date Filter: {date_specific_input['date_filter']}")
        print(f"Status: {status_msg_date}")
        print(f"Retrieved {len(reconstructed_context_date)} reconstructed Murlis.")
        
        if reconstructed_context_date:
            print("Full Murli content preview:")
            print(reconstructed_context_date[0][0]) 
            print(f"Total length of reconstructed Murli: {len(reconstructed_context_date[0][0])} characters.")
        else:
            print("Murli for the specified date was not found. Ensure '1969-01-23' with lang 'en' is in your test DB.")

        # --- Test 2: Topical query (should use hybrid search and sentence window) ---
        print("\n--- Testing Topical (Hybrid Search) Retrieval ---")
        topical_query = "What did Baba say about effort required to achieve angelic stage?"
        topical_input = {"query": topical_query, "language": "en"} # No date filter, so it uses hybrid search
        
        # Also call .func for topical query to get the full artifact
        status_msg_topical, reconstructed_context_topical = murli_tool.func(
            query=topical_input["query"],
            date_filter=topical_input.get("date_filter"), # Use .get for optional parameters
            language=topical_input["language"]
        )
        
        print(f"Query: '{topical_query}'")
        print(f"Status: {status_msg_topical}")
        print(f"Retrieved {len(reconstructed_context_topical)} context windows.")
        
        if reconstructed_context_topical:
            print("First context window preview:")
            print(reconstructed_context_topical[0][0][:500] + "...")
        
    except Exception as e:
        print(f"Error during test: {e}")