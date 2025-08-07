#!/usr/bin/env python
import os
import sys
from RAG_BOT.src.logger import logger
import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
# This loads the variables from .env
load_dotenv(dotenv_path)  

# --- Configuration ---
VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH")
if not VECTOR_STORE_PATH:
    logger.error("VECTOR_STORE_PATH environment variable not set.")
    exit(1)

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
DEFAULT_K = 25
DEFAULT_SEARCH_TYPE = "similarity"

# --- Initialize Embeddings and VectorDB Connection ---
try:
    logger.info(f"Initializing embeddings model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    logger.info(f"Connecting to Chroma DB at: {VECTOR_STORE_PATH}")
    vectordb = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
    logger.info("Chroma DB connection successful.")
except Exception as e:
    logger.error(f"Failed to initialize embeddings or connect to Chroma DB: {e}", exc_info=True)
    exit(1)

# --- MCP Server ---
mcp = FastMCP("murli-retriever-server", "0.1.0")

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

@mcp.tool()
def retrieve_murli_extracts(query: str,                             
                            date_filter: str = None,
                            language: str = "en") -> str:
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
        A string containing the relevant context snippets retrieved from the murli database, formatted as a
        coherent response. If no relevant documents are found, an empty string is returned.
    """
    logger.info(f"Executing retrieval: query='{query}', k={DEFAULT_K}, date_filter='{date_filter}'," 
                f"search_type='{DEFAULT_SEARCH_TYPE}', language='{language}'")

    try:        
        if date_filter:
            try:
                search_kwargs_base, active_formatted_date = _prepare_filters(date_filter, language)
                search_kwargs_base["k"] = DEFAULT_K
            except ValueError:
                raise ValueError("Invalid date format for date_filter. Use YYYY-MM-DD.")

        # Perform retrieval
        retriever = vectordb.as_retriever(search_type=DEFAULT_SEARCH_TYPE, search_kwargs=search_kwargs_base)
        retrieved_docs = retriever.invoke(query.strip().lower())  # Normalize query

        # Format results
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        logger.info(f"Retrieved {len(retrieved_docs)} documents with k={DEFAULT_K} and search_type='{DEFAULT_SEARCH_TYPE}'")
        return context

    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        raise RuntimeError(f"An error occurred during retrieval: {str(e)}")

if __name__ == "__main__":
    print("Starting Murli MCP Server...")
    mcp.run(transport='streamable-http')
    print("Murli MCP Server is running.")