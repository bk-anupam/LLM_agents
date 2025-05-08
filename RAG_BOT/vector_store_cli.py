# /home/bk_anupam/code/LLM_agents/RAG_BOT/vector_store_cli.py
import sys
import os

# Add the project root to the Python path to allow imports from RAG_BOT
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from RAG_BOT.logger import logger
from RAG_BOT.config import Config
from RAG_BOT.vector_store import VectorStore
from RAG_BOT.document_indexer import DocumentIndexer
from RAG_BOT.file_manager import FileManager


def test_query_index():
    """
    Test querying the index.
    """
    # Initialize with the actual VectorStore class
    vs = VectorStore(persist_directory=Config().VECTOR_STORE_PATH) 
    query = "दूसरों की चेकिंग करने के बारे में बाबा ने मुरली में क्या बताया है?"
    test_date = "1992-09-24" 
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


def index_data():
    """
    Build the index for all PDFs in 'english' and HTM files in 'hindi' subdirectories
    of the configured DATA_PATH.
    """
    config = Config()
    data_dir = config.DATA_PATH
    if not data_dir:
        logger.error("DATA_PATH is not set in the configuration. Cannot start indexing.")
        return

    logger.info(f"Starting MANUAL indexing process for base directory: {data_dir}")

    # Initialize components
    # These will use the actual classes once vector_store.py is refactored
    vector_store_instance = VectorStore(persist_directory=config.VECTOR_STORE_PATH)
    file_manager_instance = FileManager()
    document_indexer_instance = DocumentIndexer(vector_store_instance, file_manager_instance)
    
    document_indexer_instance.index_directory(data_dir)

    logger.info("Manual indexing process complete.")
    # Optionally log metadata
    vector_store_instance.log_all_indexed_metadata()


if __name__ == "__main__":
    # Example: Choose to index data or test a query.
    # For a real CLI, you might use argparse here to select actions.
    
    # To run indexing:
    # index_data()

    # To test querying (ensure data is indexed first):
    # test_query_index()

    # To log all metadata (ensure data is indexed first):
    vs = VectorStore()
    vs.log_all_indexed_metadata()
