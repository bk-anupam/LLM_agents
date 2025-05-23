import datetime
import datetime
from langchain_core.tools import tool
from langchain_chroma import Chroma
from typing import Optional, Dict, Any, Callable, List, Tuple 
from RAG_BOT.logger import logger
from RAG_BOT.config import Config


# Factory function to create the tool with the vectordb instance enclosed
def create_context_retriever_tool(vectordb: Chroma, config: Config) -> Callable:
    """
    Factory function that creates and returns the murli retriever tool.
    The returned tool function has the vectordb instance enclosed via closure.

    Args:
        vectordb: The initialized Chroma vector database instance.
        config: Config object containing retrieval parameters.
    Returns:
        A callable tool function suitable for LangChain/LangGraph.
    """

    def _prepare_filters(date_filter, language):
        base_filter_conditions = []
        active_formatted_date = None
        if date_filter:
            try:
                filter_date_obj = datetime.datetime.strptime(date_filter, '%Y-%m-%d')
                active_formatted_date = filter_date_obj.strftime('%Y-%m-%d')
                date_condition = {"date": active_formatted_date}
                base_filter_conditions.append(date_condition)
                logger.info(f"Date filter condition prepared: {date_condition}")
            except ValueError:
                logger.warning(
                    f"Invalid date format '{date_filter}'. Date filter will not be used for semantic search, "
                    "but might be attempted for 'get_all_for_date' if format is corrected by then or if language is primary."
                )
        if language:
            lang_condition = {"language": language.lower()}
            base_filter_conditions.append(lang_condition)
            logger.info(f"Language filter condition prepared: {lang_condition}")

        search_kwargs_base = {}
        if base_filter_conditions:
            if len(base_filter_conditions) > 1:
                search_kwargs_base["filter"] = {"$and": base_filter_conditions}
            else:
                search_kwargs_base["filter"] = base_filter_conditions[0]
            logger.info(f"Base filter for Chroma: {search_kwargs_base.get('filter')}")
        return search_kwargs_base, active_formatted_date

    def _execute_retrieval_with_fallback(
        vectordb_instance: Chroma,
        query_str: str,
        base_search_args: dict,
        initial_k_value: int,
        fallback_k_value: int,
        search_operation_type: str,
        description: str # For logging, e.g., "semantic search" or "date-filtered search"
    ) -> List[Any]: # Assuming Document type, but using Any for broader compatibility
        """
        Executes a retrieval operation with a fallback mechanism for HNSW runtime errors.
        Args:
            vectordb_instance: The Chroma vector database instance.
            query_str: The query string.
            base_search_args: Base keyword arguments for the search (e.g., filters).
            initial_k_value: The initial number of documents to retrieve.
            fallback_k_value: The fallback number of documents if the initial attempt fails.
            search_operation_type: The type of search (e.g., "similarity").
            description: A description of the search operation for logging.
        Returns:
            A list of retrieved documents.
        """
        retrieved_documents = []
        current_search_args = base_search_args.copy()
        current_search_args["k"] = initial_k_value
        logger.debug(f"{description.capitalize()}: Using search_kwargs: {current_search_args}")

        try:
            retriever = vectordb_instance.as_retriever(
                search_type=search_operation_type,
                search_kwargs=current_search_args
            )
            retrieved_documents = retriever.invoke(query_str)
            logger.info(f"{description.capitalize()} retrieved {len(retrieved_documents)} docs with k={initial_k_value}.")
        except RuntimeError as e_rt:
            if "Cannot return the results in a contigious 2D array" in str(e_rt):
                logger.warning(f"HNSW RuntimeError with k={initial_k_value} for {description}: {e_rt}. Attempting fallback.")
                if not (initial_k_value == 1 and fallback_k_value == 1): # Avoid infinite loop if k=1 and fallback_k=1
                    fallback_search_args = base_search_args.copy()
                    fallback_search_args["k"] = fallback_k_value
                    logger.info(f"{description.capitalize()} fallback: Using search_kwargs: {fallback_search_args}")
                    try:
                        retriever_fallback = vectordb_instance.as_retriever(
                            search_type=search_operation_type,
                            search_kwargs=fallback_search_args
                        )
                        retrieved_documents = retriever_fallback.invoke(query_str)
                        logger.info(f"{description.capitalize()} fallback successful with k={fallback_k_value}, retrieved {len(retrieved_documents)} docs.")
                    except Exception as fallback_e:
                        logger.error(f"Error during {description} fallback with k={fallback_k_value}: {fallback_e}", exc_info=True)
                else:
                    logger.error(f"HNSW RuntimeError occurred with k=1 for {description}. Cannot reduce k further. Error: {e_rt}", exc_info=True)
            else:
                logger.error(f"Unhandled RuntimeError during {description}: {e_rt}", exc_info=True)
        except Exception as e_gen:
            logger.error(f"General error during {description}: {e_gen}", exc_info=True)
        return retrieved_documents


    def _perform_semantic_search(vectordb_instance, normalized_query, search_kwargs_base, k, search_type, k_fallback):
        """
        Performs a semantic search on the provided vector database with specified parameters.
        """
        return _execute_retrieval_with_fallback(
            vectordb_instance=vectordb_instance,
            query_str=normalized_query,
            base_search_args=search_kwargs_base,
            initial_k_value=k,
            fallback_k_value=k_fallback,
            search_operation_type=search_type,
            description="semantic search"
        )


    def _perform_date_filtered_search(vectordb, normalized_query, active_formatted_date, language, k_fallback):
        """
        Perform a date-filtered search to retrieve all chunks for a specific date and language.
        """
        retrieved_docs_all_for_date = []
        max_chunks_for_date_filter = config.MAX_CHUNKS_FOR_DATE_FILTER
        # Calculate a fallback k for date-filtered search, similar to semantic search's k_fallback        
        date_filter_k_fallback = max(1, min(k_fallback, max_chunks_for_date_filter // 2 if max_chunks_for_date_filter > 1 else 1))

        if active_formatted_date:
            logger.info(f"Valid date filter '{active_formatted_date}' is active. Attempting to retrieve all chunks "
                        f"for this date (and language if any), up to {max_chunks_for_date_filter}.")
                        
            get_all_filter_conditions = [{"date": active_formatted_date}]
            if language:
                get_all_filter_conditions.append({"language": language.lower()})
            if len(get_all_filter_conditions) > 1:
                get_all_filter_chroma = {"$and": get_all_filter_conditions}
            else:
                get_all_filter_chroma = get_all_filter_conditions[0]
            if get_all_filter_chroma:
                # Note: The 'query' for this type of retrieval might be less important if the filter is very specific.
                # Using normalized_query as it's available.
                base_search_args_for_date = {"filter": get_all_filter_chroma}
                retrieved_docs_all_for_date = _execute_retrieval_with_fallback(
                    vectordb_instance=vectordb,
                    query_str=normalized_query, # Or a generic query if filters are primary
                    base_search_args=base_search_args_for_date,
                    initial_k_value=max_chunks_for_date_filter,
                    fallback_k_value=date_filter_k_fallback,
                    search_operation_type=config.SEARCH_TYPE, # Usually 'similarity' even for filtered retrieval
                    description="date-filtered search"
                )
                logger.info(f"Retrieved {len(retrieved_docs_all_for_date)} additional/overlapping chunks specifically "
                            f"for date '{active_formatted_date}' (limit {max_chunks_for_date_filter}).")
            else:
                logger.info("No valid filter for 'get_all_for_date' step, skipping.")
        return retrieved_docs_all_for_date


    def _combine_and_deduplicate(retrieved_docs_semantic, retrieved_docs_all_for_date):
        """
        Combines two lists of retrieved document objects, deduplicating them based on their 'page_content' attribute.

        Args:
            retrieved_docs_semantic (list): List of document objects retrieved semantically.
            retrieved_docs_all_for_date (list): List of document objects retrieved by date.

        Returns:
            list: A deduplicated list of document objects, preserving only one instance per unique 'page_content'.

        Logs:
            Issues a warning if a document object does not have a 'page_content' attribute.
        """
        combined_docs_map = {}
        for doc in retrieved_docs_semantic + retrieved_docs_all_for_date:
            if hasattr(doc, 'page_content'):
                combined_docs_map[doc.page_content] = doc
            else:
                logger.warning(f"Retrieved item without page_content: {doc}")
        return list(combined_docs_map.values())


    @tool(response_format="content_and_artifact")
    def retrieve_context(
        query: str,
        date_filter: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        """
        Retrieves relevant context snippets from indexed Brahmakumaris murlis based on a user query,
        optionally filtering by date and language. Use this tool when the user asks for summaries,
        details, specific points, or content related to a particular date (YYYY-MM-DD), topic,
        or concept mentioned within the murlis. The tool accesses an underlying vector database
        containing the murli content.

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
            1. A status string indicating the outcome (e.g., number of documents retrieved).
            2. A list of strings, where each string is the page content of a retrieved document chunk.
            Returns ("Error retrieving context.", []) if an error occurs.
        """
        k = config.INITIAL_RETRIEVAL_K
        search_type = config.SEARCH_TYPE
        k_fallback = getattr(config, "K_FALLBACK", max(1, min(5, k // 2 if k > 1 else 1)))
        max_chunks_for_date_filter = config.MAX_CHUNKS_FOR_DATE_FILTER

        logger.info(f"Executing context_retriever_tool for query: '{query}', date: {date_filter}, lang: {language}, K: {k}, "
                    f"search_type: {search_type}, max_chunks_for_date: {max_chunks_for_date_filter}")

        normalized_query = query.strip().lower()
        search_kwargs_base, active_formatted_date = _prepare_filters(date_filter, language)
        retrieved_docs_semantic = _perform_semantic_search(
            vectordb, normalized_query, search_kwargs_base, k, search_type, k_fallback
        )
        if active_formatted_date:            
            retrieved_docs_all_for_date = _perform_date_filtered_search(
                vectordb, normalized_query, active_formatted_date, language, k_fallback
            )
            final_docs_to_process = _combine_and_deduplicate(retrieved_docs_semantic, retrieved_docs_all_for_date)
        else:
            final_docs_to_process = retrieved_docs_semantic

        if not final_docs_to_process:
            status_message = "No relevant documents found matching the criteria after all retrieval attempts."
            logger.info(status_message)
            return status_message, []

        doc_contents = [doc.page_content for doc in final_docs_to_process]

        if active_formatted_date and retrieved_docs_all_for_date:
            status_message = (
                f"Retrieved {len(retrieved_docs_semantic)} docs via semantic search "
                f"and augmented with {len(retrieved_docs_all_for_date)} docs by targeting date '{active_formatted_date}'. "
                f"Total unique docs for reranking: {len(final_docs_to_process)}."
            )
        else:
            status_message = f"Retrieved {len(final_docs_to_process)} unique document chunks for reranking (semantic search only)."

        logger.info(status_message)
        if doc_contents:
            logger.info(f"First combined doc content snippet for reranker: {doc_contents[0][:200]}...")

        return status_message, doc_contents

    return retrieve_context


# Example usage (for testing purposes, requires a Chroma instance)
if __name__ == '__main__':
    # This is a placeholder for testing. Replace with actual Chroma DB setup.
    from langchain_huggingface import HuggingFaceEmbeddings
    from RAG_BOT.config import Config 
    persist_directory = '/home/bk_anupam/code/LLM_agents/RAG_BOT/chroma_db' # Example path
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vectordb_instance = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        test_query = "What is soul consciousness?"

        # 1. Create the tool using the factory, passing config values
        config_instance = Config()
        murli_tool_instance = create_context_retriever_tool(
            vectordb=vectordb_instance,
            config=config_instance
        )
        logger.info("Test: Context retriever tool created using config instance.")

        # 2. Invoke the created tool
        # The tool now only expects arguments defined in its signature (query, date_filter)
        # k and search_type are now part of the tool's internal configuration via the factory
        tool_input = {"query": test_query} # Removed k from here
        context_result_list = murli_tool_instance.invoke(tool_input)

        print(f"Query: {test_query}")
        print(f"Retrieved {len(context_result_list)} Context Documents (using k={Config.K}, search_type='{Config.SEARCH_TYPE}'):")
        if context_result_list:
            print("First document content:")
            print(context_result_list[0][:500] + "...") # Print snippet of first doc
        else:
            print("No documents retrieved.")
        print("\nFactory function defined. Tool created and tested.")

    except Exception as e:
        print(f"Error during test setup or execution: {e}")
    # pass # Keep pass if not running example directly
