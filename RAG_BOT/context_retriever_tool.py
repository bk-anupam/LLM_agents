import datetime
from langchain_core.tools import tool
from langchain_chroma import Chroma
from RAG_BOT.logger import logger
from typing import Optional, Dict, Any, Callable

# Factory function to create the tool with the vectordb instance enclosed
def create_context_retriever_tool(vectordb: Chroma, k: int = 25, search_type: str = "similarity") -> Callable:
    """
    Factory function that creates and returns the murli retriever tool.
    The returned tool function has the vectordb instance enclosed via closure.

    Args:
        vectordb: The initialized Chroma vector database instance.
        k: The number of documents to retrieve (default: 25).
        search_type: The type of search to perform ('similarity', 'mmr', etc. Default: 'similarity').

    Returns:
        A callable tool function suitable for LangChain/LangGraph.
    """

    @tool
    def retrieve_context(
        query: str,        
        date_filter: Optional[str] = None,        
    ) -> str:
        """
        Retrieves relevant context snippets from Brahmakumaris murlis stored in a Chroma vector database based on the user query.
        This tool accesses the vectordb instance provided during its creation.

        Args:
            query: The user's query string.            
            date_filter: An optional date string in 'YYYY-MM-DD' format to filter documents by date.            

        Returns:
            A string containing the concatenated page content of the retrieved documents,
            separated by double newlines. Returns an empty string if no documents are found
            or if an error occurs.
        """
        logger.info(f"Executing context_retriever_tool for query: {query}")
        try:
            # Normalize query
            normalized_query = query.strip().lower()

            # Prepare search kwargs
            search_kwargs: Dict[str, Any] = {"k": k}

            if date_filter:
                try:
                    filter_date = datetime.datetime.strptime(date_filter, '%Y-%m-%d')
                    formatted_date = filter_date.strftime('%Y-%m-%d')
                    logger.info(f"Applying date filter: {formatted_date}")
                    filter_criteria = {"date": {"$eq": formatted_date}}
                    search_kwargs["filter"] = filter_criteria
                except ValueError:
                    logger.error("Invalid date format provided to murli_retriever. Please use YYYY-MM-DD.")
                    # Ignore the filter if format is wrong.
                    pass # Or return "Error: Invalid date format."

            # Create retriever using the enclosed vectordb
            retriever = vectordb.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )

            # Retrieve documents
            retrieved_docs = retriever.invoke(normalized_query)

            # Format context
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            logger.info(f"Retrieved {len(retrieved_docs)} documents using context_retriever_tool.")
            logger.info(f"Context retrieved: {context[:200]}...")  
            return context

        except Exception as e:
            logger.error(f"Error during context retrieval: {e}", exc_info=True)
            return "Error: Could not retrieve context due to an internal issue."

    # Return the decorated inner function
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
        murli_tool_instance = create_context_retriever_tool(
            vectordb=vectordb_instance,
            k=Config.K,                 # Use K from config
            search_type=Config.SEARCH_TYPE # Use search_type from config
        )
        logger.info(f"Test: Context retriever tool created with k={Config.K} and search_type='{Config.SEARCH_TYPE}'.")

        # 2. Invoke the created tool
        # The tool now only expects arguments defined in its signature (query, date_filter)
        # k and search_type are now part of the tool's internal configuration via the factory
        tool_input = {"query": test_query} # Removed k from here
        context_result = murli_tool_instance.invoke(tool_input)

        print(f"Query: {test_query}")
        print(f"Retrieved Context (using k={Config.K}, search_type='{Config.SEARCH_TYPE}'):\n{context_result}")
        print("\nFactory function defined. Tool created and tested.")

    except Exception as e:
        print(f"Error during test setup or execution: {e}")
    # pass # Keep pass if not running example directly
