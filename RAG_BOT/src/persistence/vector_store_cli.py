from RAG_BOT.src.logger import logger
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.persistence.vector_store import VectorStore
from RAG_BOT.src.services.document_indexer import DocumentIndexer
from RAG_BOT.src.file_manager import FileManager


def test_query_index():
    """
    Test querying the index.
    """
    config = Config() # Create a config instance
    # Initialize with the actual VectorStore class
    vs = VectorStore(persist_directory=config.VECTOR_STORE_PATH, config=config)
    # query = "दूसरों की चेकिंग करने के बारे में बाबा ने मुरली में क्या बताया है?"
    # query = "भट्ठी की जो सारी पढ़ाई वा शिक्षा ली उसका सार कौन से तीन शब्दों में याद रखने के लिए बाबा ने मुरलियों में जोर दिया"
    query = "2025-09-10 की मुरली की क्या शिक्षाएं हैं?"
    # query = "What did Baba say about checking others in the Murlis?"
    test_date = "2025-09-10"
    language = "hi"
    logger.info(f"Testing query: {query} with date filter: {test_date} and language: {language}")
    try:
        result = vs.query_index(query, language=language, date_filter=test_date)        
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
    vector_store_instance = VectorStore(persist_directory=config.VECTOR_STORE_PATH, config=config)
    file_manager_instance = FileManager(config=config)
    document_indexer_instance = DocumentIndexer(
        vector_store_instance=vector_store_instance,
        file_manager_instance=file_manager_instance,
        config=config
    )
    
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
    test_query_index()

    # To log all metadata (ensure data is indexed first):
    # config_instance_for_logging = Config()
    # vs = VectorStore(config=config_instance_for_logging)
    # vs.log_all_indexed_metadata()
