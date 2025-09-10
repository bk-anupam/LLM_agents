# /home/bk_anupam/code/LLM_agents/RAG_BOT/main.py
import asyncio
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.logger import logger
from RAG_BOT.src.services.gcs_uploader import GCSUploaderService
from RAG_BOT.src.telegram.bot import TelegramBotApp
from RAG_BOT.src.persistence.vector_store import VectorStore
from RAG_BOT.src.services.document_indexer import DocumentIndexer
from RAG_BOT.src.file_manager import FileManager
from RAG_BOT.src.processing.pdf_processor import PdfProcessor
from RAG_BOT.src.processing.htm_processor import HtmProcessor
from RAG_BOT.src.json_parser import JsonParser
from RAG_BOT.src.persistence.conversation_interfaces import AbstractThreadManager
from RAG_BOT.src.persistence.firestore_thread_manager import FirestoreThreadManager
from RAG_BOT.src.utils import get_checkpointer

def get_thread_manager(config: Config) -> AbstractThreadManager:
    """
    Factory function to get the appropriate thread manager based on configuration.
    """
    backend = config.CONV_PERSISTENCE_BACKEND.lower()
    logger.info(f"Creating thread manager with backend: '{backend}'")

    if backend == "firestore":
        return FirestoreThreadManager(project_id=config.GCP_PROJECT_ID)
    elif backend == "sqlite":
        # This part is ready for when we implement SQLiteThreadManager
        pass
    else:
        raise ValueError(f"Unsupported persistence backend in config: '{backend}'")


def main_setup_and_run():
    """
    Orchestrates the setup and execution of the RAG Bot application.
    Initializes configuration, services, and the bot application,
    then starts the bot.
    """
    bot_app = None  
    try:
        config = Config()
        # Get checkpointer and thread manager from factories based on config
        checkpointer = get_checkpointer(config)
        thread_manager = get_thread_manager(config)

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
        gcs_uploader = GCSUploaderService(config)

        # Create and run the bot application
        bot_app = TelegramBotApp(
            config=config,
            vector_store_instance=vector_store_instance,
            pdf_processor=pdf_processor,
            htm_processor=htm_processor,
            thread_manager=thread_manager,
            gcs_uploader=gcs_uploader
        )

        json_parser = JsonParser()
        logger.info("Submitting async initialization to bot's event loop...")        
        # asyncio.run_coroutine_threadsafe is the bridge between the synchronous main thread and the asynchronous 
        # event loop running in bot.py background thread. This function is specifically designed to be called from 
        # a different thread than the one the asyncio event loop is running in.
        # It takes the async function we want to run (bot_app.initialize_agent_and_handler(...)) and safely submits 
        # it to the event loop (bot_app.loop) that is running in the background thread.
        # Crucially, this call is non-blocking. It returns immediately with a concurrent.futures.Future object, 
        # which is a placeholder for the eventual result of the async function.
        future = asyncio.run_coroutine_threadsafe(
            bot_app.initialize_agent_and_handler(config=config, checkpointer=checkpointer, json_parser=json_parser),
            bot_app.loop)
        # blocks the main thread and waits for the Future object to be populated with a result. 
        future.result(timeout=config.ASYNC_OPERATION_TIMEOUT)
        logger.info("Async initialization of agent and handler complete.")

        bot_app.run()

    except KeyboardInterrupt:
        logger.info("Shutdown signal (KeyboardInterrupt) received. Stopping bot...")
    except Exception as e:
        logger.critical(f"An unhandled exception occurred during application lifecycle: {str(e)}", exc_info=True)
    finally:
        if bot_app:
            logger.info("Initiating graceful shutdown...")
            bot_app.stop()
        logger.info("Application has been shut down.")

if __name__ == "__main__":
    main_setup_and_run()