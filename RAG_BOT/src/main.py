# /home/bk_anupam/code/LLM_agents/RAG_BOT/main.py
import asyncio
from google.cloud import firestore
from RAG_BOT.src.persistence.firestore_checkpointer import AsyncFirestoreSaver
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.logger import logger
from RAG_BOT.src.telegram.bot import TelegramBotApp
from RAG_BOT.src.persistence.vector_store import VectorStore
from RAG_BOT.src.services.document_indexer import DocumentIndexer
from RAG_BOT.src.file_manager import FileManager
from RAG_BOT.src.processing.pdf_processor import PdfProcessor
from RAG_BOT.src.processing.htm_processor import HtmProcessor
from RAG_BOT.src.json_parser import JsonParser

def main_setup_and_run():
    """
    Orchestrates the setup and execution of the RAG Bot application.
    Initializes configuration, services, and the bot application,
    then starts the bot.
    """
    try:
        config = Config()
        # Initialize Firestore client
        db = firestore.AsyncClient(project=config.GCP_PROJECT_ID, database="rag-bot-firestore-db")
        checkpointer = AsyncFirestoreSaver(db)
        logger.info("AsyncFirestoreSaver checkpointer initialized successfully.")

        DATA_DIRECTORY = config.DATA_PATH
        logger.info(f"Data directory set to: {DATA_DIRECTORY}")

        # Initialize persistence layer
        vector_store_instance = VectorStore(persist_directory=config.VECTOR_STORE_PATH, config=config)
        vectordb = vector_store_instance.get_vectordb()
        logger.info("VectorStore initialized.")

        if vectordb is None:
            logger.error("VectorDB instance is None. Cannot proceed.")
            exit(1)

        # Index data on startup if configured
        if config.INDEX_ON_STARTUP:
            file_manager_instance = FileManager(config=config)
            document_indexer_instance = DocumentIndexer(
                vector_store_instance=vector_store_instance,
                file_manager_instance=file_manager_instance,
                config=config
            )
            document_indexer_instance.index_directory(DATA_DIRECTORY)

        logger.info("Logging final indexed metadata...")
        vector_store_instance.log_all_indexed_metadata()

        # Initialize processing layer
        pdf_processor = PdfProcessor()
        htm_processor = HtmProcessor()

        # Create and run the bot application
        bot_app = TelegramBotApp(
            config=config,
            vector_store_instance=vector_store_instance,
            pdf_processor=pdf_processor,
            htm_processor=htm_processor
        )

        json_parser = JsonParser()        
        # Asynchronously initialize the agent and message handler
        async def _async_init_for_bot_app():
            await bot_app.initialize_agent_and_handler(
                vectordb, 
                config=config, 
                checkpointer=checkpointer, 
                json_parser=json_parser
            )

        logger.info("Submitting async initialization to bot's event loop...")
        future = asyncio.run_coroutine_threadsafe(_async_init_for_bot_app(), bot_app.loop)
        
        future.result(timeout=config.ASYNC_OPERATION_TIMEOUT)
        logger.info("Async initialization of agent and handler complete.")

        bot_app.run()

    except Exception as e:
        logger.critical(f"Failed during application startup: {str(e)}", exc_info=True)
        # In a real-world scenario, you might want to ensure the bot's event loop is stopped cleanly
        exit(1)

if __name__ == "__main__":
    main_setup_and_run()
