import datetime
import datetime
from langchain_core.tools import tool
from langchain_chroma import Chroma
from typing import Optional, Dict, Any, Callable, List, Tuple 
import os
import sys
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from RAG_BOT.logger import logger


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
            query: The core semantic query about the murli content (e.g., 'summary of teachings', 'points about effort'). The LLM should formulate this based on the user's original question.
            date_filter: An optional date string in 'YYYY-MM-DD' format extracted from the user's query to filter documents by date. Provide ONLY if a specific date is mentioned.
            language: The language code ('en' for English, 'hi' for Hindi) inferred from the user's original query. ALWAYS try to infer and provide this parameter based on the user's input language.

        Returns:            
            A tuple containing:
            1. A status string indicating the outcome (e.g., number of documents retrieved).
            2. A list of strings, where each string is the page content of a retrieved document chunk.
            Returns ("Error retrieving context.", []) if an error occurs.
        """
        logger.info(f"Executing context_retriever_tool for query: '{query}', date: {date_filter}, lang: {language}")
        try:
            # Normalize query
            normalized_query = query.strip().lower()
            # Prepare search kwargs
            search_kwargs: Dict[str, Any] = {"k": k}
            if date_filter:
                try:
                    # Validate and format date
                    filter_date = datetime.datetime.strptime(date_filter, '%Y-%m-%d')
                    formatted_date = filter_date.strftime('%Y-%m-%d')                    
                    # Use $and if language filter is also present, otherwise just date
                    date_condition = {"date": formatted_date} # Chroma uses implicit $eq
                    if language:
                        lang_condition = {"language": language.lower()}
                        search_kwargs["filter"] = {"$and": [date_condition, lang_condition]}
                        logger.info(f"Applying date filter: {formatted_date}, language filter: {language.lower()}")
                    else:
                        search_kwargs["filter"] = date_condition
                except ValueError:
                    logger.warning(f"Invalid date format '{date_filter}'. Ignoring date filter.")
                    # If only language is present after invalid date
                    if language:
                         search_kwargs["filter"] = {"language": language.lower()}
            elif language: # Only language filter is present
                logger.info(f"Applying language filter: {language.lower()}")
                search_kwargs["filter"] = {"language": language.lower()}

            logger.debug(f"Using search_kwargs: {search_kwargs}")
            # Create retriever using the enclosed vectordb
            retriever = vectordb.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )

            # Retrieve documents
            retrieved_docs = retriever.invoke(normalized_query)
            # Return list of document contents
            doc_contents = [doc.page_content for doc in retrieved_docs]

            if not doc_contents:
                logger.info("No documents found matching the query.")
                content_string = "No relevant documents found matching the criteria."
            else:
                content_string = f"Successfully retrieved {len(retrieved_docs)} document chunks based on the query and filters."
                logger.info(f"Retrieved {len(doc_contents)} chunks.")
                logger.info(f"First retrieved doc content snippet: {doc_contents[0][:200]}...")

            return content_string, doc_contents

        except Exception as e:
            logger.error(f"Error during context retrieval: {e}", exc_info=True)
            return "Error during context retrieval.", [] # Return error status and empty list

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
