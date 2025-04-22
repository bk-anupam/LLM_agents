#!/usr/bin/env python
import os
import sys
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from RAG_BOT.logger import logger
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

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
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

@mcp.tool()
def retrieve_murli_extracts(query: str, k: int = DEFAULT_K, date_filter: str = None, 
                            search_type: str = DEFAULT_SEARCH_TYPE) -> str:
    """Retrieves relevant extracts from Brahmakumaris murlis based on a query."""
    logger.info(f"Executing retrieval: query='{query}', k={k}, date_filter='{date_filter}', search_type='{search_type}'")

    try:
        # Prepare search arguments
        extra_kwargs = {"k": k}
        if date_filter:
            try:
                filter_date = datetime.datetime.strptime(date_filter, '%Y-%m-%d')
                formatted_date = filter_date.strftime('%Y-%m-%d')
                logger.info(f"Applying date filter: {formatted_date}")
                filter_criteria = {"date": {"$eq": formatted_date}}  # Assuming 'date' metadata exists
                extra_kwargs["filter"] = filter_criteria
            except ValueError:
                raise ValueError("Invalid date format for date_filter. Use YYYY-MM-DD.")

        # Perform retrieval
        retriever = vectordb.as_retriever(search_type=search_type, search_kwargs=extra_kwargs)
        retrieved_docs = retriever.invoke(query.strip().lower())  # Normalize query

        # Format results
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        logger.info(f"Retrieved {len(retrieved_docs)} documents.")
        return context

    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        raise RuntimeError(f"An error occurred during retrieval: {str(e)}")

if __name__ == "__main__":
    print("Starting Murli MCP Server...")
    mcp.run(transport='stdio')
    print("Murli MCP Server is running.")