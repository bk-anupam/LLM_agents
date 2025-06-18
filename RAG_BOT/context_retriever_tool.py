from langchain_core.tools import tool
from langchain_chroma import Chroma
from typing import Optional, Dict, Any, Callable, List, Tuple 
from RAG_BOT.logger import logger
from RAG_BOT.config import Config
from RAG_BOT.context_retrieval.filter_processor import FilterProcessor
from RAG_BOT.context_retrieval.bm25_processor import BM25Processor
from RAG_BOT.context_retrieval.result_processor import ResultProcessor


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
            2. A list of (full_murli_content, metadata) tuples for reconstructed Murlis.
            Returns ("Error retrieving context.", []) if an error occurs.
        """
        # Configuration
        k_semantic = config.INITIAL_RETRIEVAL_K
        k_bm25 = config.BM25_TOP_K
        max_corpus_for_bm25 = config.BM25_MAX_CORPUS_SIZE
        search_type = config.SEARCH_TYPE
        k_fallback = getattr(config, "K_FALLBACK", max(1, min(5, k_semantic // 2 if k_semantic > 1 else 1)))
        
        logger.info(f"Executing Murli retrieval for query: '{query}', date: {date_filter}, lang: {language}")
        
        # Prepare filters
        search_kwargs, active_date = FilterProcessor.prepare_filters(date_filter, language)
        normalized_query = query.strip().lower()
        
        # 1. Semantic Search
        semantic_docs = RetrieverExecutor.execute_with_fallback(
            vectordb, normalized_query, search_kwargs, k_semantic, k_fallback, search_type, "semantic search"
        )
        logger.info(f"Semantic search retrieved {len(semantic_docs)} chunks.")
        # to delete
        # log all retrieved documents
        if semantic_docs:
            logger.debug(f"Retrieved {len(semantic_docs)} semantic documents: {[doc.page_content for doc in semantic_docs]}")
        
        # 2. BM25 Search (if filters present)
        bm25_results = []
        if search_kwargs.get("filter"):
            logger.info("Filters present, proceeding with BM25 search.")
            corpus_items = bm25_processor.get_scoped_corpus(search_kwargs["filter"], max_corpus_for_bm25)
            if corpus_items:
                bm25_results = BM25Processor.search(normalized_query, corpus_items, k_bm25)
                logger.info(f"BM25 search retrieved {len(bm25_results)} chunks.")
                # to delete
                # Log content of BM25 results
                logger.debug(f"BM25 results: {[item[0] for item in bm25_results]}")  
        
        # 3. Combine results
        combined_chunks = result_processor.combine_and_deduplicate(semantic_docs, bm25_results)
        
        if not combined_chunks:
            status_msg = "No relevant chunks found from hybrid retrieval."
            logger.info(status_msg)
            return status_msg, []        
        
        # 4. Return chunks or reconstruct Murlis based on flag
        if not config.RECONSTRUCT_MURLIS:
            status_msg = (
                f"Retrieved {len(semantic_docs)} semantic + {len(bm25_results)} BM25 chunks. "
                f"Returning {len(combined_chunks)} unique chunks (no reconstruction)."
            )
            logger.info(status_msg)
            return status_msg, combined_chunks
        
        # 4. Reconstruct Murlis
        logger.info(f"Total {len(combined_chunks)} unique chunks for Murli reconstruction.")
        chunk_metadatas = [meta for _, meta in combined_chunks]
        reconstructed_murlis = result_processor.reconstruct_murlis(chunk_metadatas, vectordb, config)
        
        if not reconstructed_murlis:
            status_msg = "Failed to reconstruct any full Murlis from chunks."
            logger.info(status_msg)
            return status_msg, []
        
        # Generate status message
        status_msg = (
            f"Retrieved {len(semantic_docs)} semantic + {len(bm25_results)} BM25 chunks. "
            f"Reconstructed {len(reconstructed_murlis)} full Murlis."
        )
        logger.info(status_msg)
        
        if reconstructed_murlis:
            logger.info(f"First reconstructed Murli snippet: {reconstructed_murlis[0][0][:200]}...")
        
        return status_msg, reconstructed_murlis
    
    return retrieve_context


# Example usage
if __name__ == '__main__':
    from langchain_huggingface import HuggingFaceEmbeddings
    from RAG_BOT.config import Config
    
    persist_directory = '/home/bk_anupam/code/LLM_agents/RAG_BOT/chroma_db'
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        vectordb_instance = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        config_instance = Config()
        
        murli_tool = create_context_retriever_tool(vectordb_instance, config_instance)
        logger.info("Context retriever tool created successfully.")
        
        # Test the tool
        test_query = "What is soul consciousness?"
        tool_input = {"query": test_query, "date_filter": "2023-01-01"}
        result = murli_tool.invoke(tool_input)
        
        print(f"Query: {test_query}")
        print(f"Status: {result[0]}")
        print(f"Retrieved {len(result[1])} reconstructed Murlis")
        
        if result[1]:
            print("First Murli content preview:")
            print(result[1][0][0][:500] + "...")
        
    except Exception as e:
        print(f"Error during test: {e}")