import telebot
import asyncio
from telebot.types import Update
import threading # Added for dedicated event loop
import os
from flask import Flask, request, jsonify
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.json_parser import JsonParser
from RAG_BOT.src.logger import logger
from RAG_BOT.src.persistence.vector_store import VectorStore
from RAG_BOT.src.agent.graph_builder import build_agent
from RAG_BOT.src.services.message_processor import MessageProcessor
from RAG_BOT.src.processing.pdf_processor import PdfProcessor
from RAG_BOT.src.processing.htm_processor import HtmProcessor 
from langgraph.checkpoint.base import BaseCheckpointSaver
from RAG_BOT.src.persistence.user_settings_manager import UserSettingsManager
from RAG_BOT.src.persistence.update_manager import UpdateManager
from RAG_BOT.src.persistence.conversation_interfaces import AbstractThreadManager
from RAG_BOT.src.telegram.handlers.handler_registry import HandlerRegistry


class TelegramBotApp:
    """
    The main application class that sets up the Telegram bot, Flask app,
    and integrates all components. This class is the orchestrator.
    """
    def __init__(
        self, 
        config: Config, 
        vector_store_instance: VectorStore, 
        pdf_processor: PdfProcessor = None, 
        htm_processor: HtmProcessor = None, 
        thread_manager: AbstractThreadManager = None
    ):
        # Initialize Flask app
        self.app = Flask(__name__)
        self.config = config
        self.thread_manager = thread_manager
        self.user_settings_manager = UserSettingsManager(project_id=config.GCP_PROJECT_ID)
        
        # Initialize the UpdateManager to handle duplicate messages
        self.update_manager = UpdateManager(project_id=config.GCP_PROJECT_ID)
        self.update_manager.cleanup_old_updates() 
        
        self.agent = None
        self.handler_registry = None # Placeholder for the handler registry

        self.vector_store_instance = vector_store_instance
        self.pdf_processor = pdf_processor or PdfProcessor()
        self.htm_processor = htm_processor or HtmProcessor()         
        self._rag_bot_package_dir = os.path.abspath(os.path.dirname(__file__))
        self.project_root_dir = os.path.abspath(os.path.join(self._rag_bot_package_dir, '..'))
        
        # Setup asyncio event loop for asynchronous tasks and agent invocations
        # This event loop will run in a dedicated thread to avoid blocking the main thread
        # (which handles incoming Telegram webhooks using Flask and pyTelegramBotAPI synchronously).
        self.loop = asyncio.new_event_loop()        
        # Make the thread non-daemon to prevent abrupt termination. A non-daemon thread (like this one) 
        # will prevent the Python program from exiting until this thread has completed its execution.
        self.thread = threading.Thread(target=self._run_loop, daemon=False)
        self.thread.start()        
        # Initialize Telegram bot and handlers
        self._initialize_telegram_bot()
        

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        logger.info(f"Asyncio event loop starting in thread: {threading.current_thread().name}")
        try:
            # This is the core of the background event loop. It starts the event loop and keeps it running indefinitely 
            # until self.loop.stop() is explicitly called (which happens during the application's shutdown). 
            # This call is blocking within this specific thread, meaning this thread dedicates itself to running the 
            # event loop and processing scheduled asynchronous tasks.
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"Asyncio event loop encountered an error: {e}", exc_info=True)
        finally:
            logger.info("Asyncio event loop is stopping...")
            self.loop.close()
            logger.info("Asyncio event loop has been closed.")


    async def initialize_agent_and_handler(
        self,
        vectordb,
        config: Config,
        checkpointer: BaseCheckpointSaver = None,
        json_parser: JsonParser = None
    ):
        """Initializes RAG agent and MessageHandler using the app's dedicated event loop."""
        logger.info("Initializing RAG agent and MessageHandler in dedicated loop...")
        # When this coroutine is run via run_coroutine_threadsafe on self.loop,
        # `await build_agent` will execute within self.loop's context.
        self.agent = await build_agent(vectordb=vectordb, config_instance=config, checkpointer=checkpointer)        
        message_processor = MessageProcessor(
            agent=self.agent, 
            config=config, 
            json_parser=json_parser, 
            thread_manager=self.thread_manager
        ) 
        logger.info("RAG agent and MessageProcessor instance created successfully.")

        # Now that the message handler is ready, initialize the command handler
        self.handler_registry = HandlerRegistry(
            bot=self.bot,
            config=self.config,
            user_settings_manager=self.user_settings_manager,
            thread_manager=self.thread_manager,
            message_processor=message_processor,
            vector_store_instance=self.vector_store_instance,
            pdf_processor=self.pdf_processor,
            htm_processor=self.htm_processor,
            loop=self.loop,
            project_root_dir=self.project_root_dir
        )
        self.handler_registry.register_handlers()
        logger.info("HandlerRegistry initialized and all handlers registered.")


    def _initialize_telegram_bot(self):
        """Initializes the Telegram bot, webhook, and message handlers."""
        if not self.config.TELEGRAM_BOT_TOKEN:
            logger.error("TELEGRAM_BOT_TOKEN is not set. Please set it in your environment variables.")
            exit(1)
        try:
            # Create Telegram bot instance
            self.bot = telebot.TeleBot(self.config.TELEGRAM_BOT_TOKEN)
            # CRITICAL FIX: Configure session management to prevent stale connections
            import telebot.apihelper as apihelper            
            # Force session recreation every 5 minutes (300 seconds) of inactivity
            # This prevents the "Remote end closed connection" issue
            apihelper.SESSION_TIME_TO_LIVE = 5 * 60  # 5 minutes            
            # Set connection timeout and retry configuration
            apihelper.CONNECT_TIMEOUT = 15  # Connection timeout in seconds
            apihelper.READ_TIMEOUT = 30     # Read timeout in seconds                                    
            logger.info("Telegram bot instance created with enhanced session management.")                        
            # Setup webhook route after initializing bot and config
            self._setup_webhook_route()
            self._setup_health_check_route() # Add health check route
            # Handler registration is now deferred until initialize_agent_and_handler is called,
            # ensuring all dependencies are ready.
        except Exception as e:
            logger.critical(f"Failed during bot initialization: {str(e)}", exc_info=True)
            exit(1)


    def _setup_webhook_route(self):
        """Sets up the webhook endpoint for Telegram."""
        @self.app.route(f'/{self.config.TELEGRAM_BOT_TOKEN}', methods=['POST'])
        def webhook():
            """Handle incoming webhook requests from Telegram"""
            if request.headers.get('content-type') == 'application/json':
                logger.info("Received webhook request") 
                try:
                    json_data = request.get_json()
                    update = Update.de_json(json_data)                    
                    # Atomically check for and mark the update as processed to prevent duplicates.
                    if self.update_manager.check_and_mark_update(update.update_id):
                        logger.info(f"Duplicate update ID {update.update_id} received, ignoring.")
                        return jsonify({"status": "ok, duplicate"})
                    # The update is new and has been marked, so we can process it.
                    self.bot.process_new_updates([update])
                    return jsonify({"status": "ok"})
                except Exception as e:
                    logger.error(f"Error processing webhook update: {e}", exc_info=True)
                    return jsonify({"status": "error", "message": "Internal server error"}), 500
            else:
                logger.warning(f"Received invalid content type for webhook: {request.headers.get('content-type')}")
                return jsonify({"status": "error", "message": "Invalid content type"}), 400


    def _setup_health_check_route(self):
        """Sets up the health check endpoint."""
        @self.app.route('/', methods=['GET'])
        def health_check():
            """Simple health check endpoint."""
            return jsonify({"status": "ok"}), 200


    # Setup and webhook configuration functions
    def setup_webhook(self, url):
        """Set up the webhook for the Telegram bot"""
        if not url:
            logger.error("WEBHOOK_URL is not configured. Cannot set webhook.")
            return False # Indicate failure
        try:
            webhook_url = f"{url.rstrip('/')}/{self.config.TELEGRAM_BOT_TOKEN}"
            logger.info("Removing existing webhook (if any)...")
            self.bot.remove_webhook()
            logger.info(f"Setting webhook to: {webhook_url}")
            success = self.bot.set_webhook(url=webhook_url)
            if success:
                logger.info("Webhook set successfully.")
                return True
            else:
                logger.error("Failed to set webhook.")
                return False
        except Exception as e:
            logger.error(f"Error setting up webhook: {e}", exc_info=True)
            return False


    def start_polling(self):
        """Starts the bot in polling mode."""
        logger.info("Removing existing webhook (if any) before starting polling...")
        self.bot.remove_webhook() # Crucial: remove webhook before polling
        logger.info("Starting bot in polling mode...")
        try:
            self.bot.infinity_polling()
        except Exception as e:
            logger.critical(f"Error during bot polling: {e}", exc_info=True)
            exit(1)
        logger.info("Bot polling stopped.")


    def run(self):
        """Runs the bot, either in polling mode or as a Flask application with webhook."""
        if not self.agent or not self.handler_registry:
            logger.critical("Core components not initialized before run(). Ensure initialize_agent_and_handler() was successfully called.")
            exit(1)

        if self.config.USE_POLLING:
            logger.info("Starting bot in polling mode as per configuration.")
            self.start_polling()
        else:
            WEBHOOK_URL = self.config.WEBHOOK_URL
            if not WEBHOOK_URL:
                logger.error("WEBHOOK_URL is not set in config. Cannot start Flask server with webhook.")
                exit(1)

            if self.setup_webhook(WEBHOOK_URL):
                logger.info(f"Starting Flask server on port {self.config.PORT}")
                self.app.run(host='0.0.0.0', port=self.config.PORT, debug=False)
            else:
                logger.critical("Failed to set up webhook. Aborting Flask server start.")
                exit(1)


    def stop(self):
        """Stops the bot and its background event loop."""
        logger.info("Stopping Telegram bot application...")
        # Stop polling if it's running
        if self.bot and hasattr(self.bot, 'stop_polling'):
            self.bot.stop_polling()
            logger.info("Bot polling stopped.")
        
        # Stop the asyncio event loop
        if self.loop.is_running():
            logger.info("Submitting stop signal to the event loop...")
            # the correct, thread-safe way to stop an asyncio event loop from a different thread. It schedules 
            # the self.loop.stop() command to be run on the event loop itself. When executed, self.loop.stop() 
            # causes the self.loop.run_forever() call (inside the _run_loop method) to finally return.
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        # Wait for the thread to finish
        logger.info("Waiting for the event loop thread to join...")
        self.thread.join(timeout=5) # Add a timeout
        if self.thread.is_alive():
            logger.warning("Event loop thread did not finish in time.")
        else:
            logger.info("Event loop thread joined successfully.")
        
        logger.info("Telegram bot application stopped.")


if __name__ == '__main__':
    # This part is for local development/testing without the full main.py setup
    # The old direct run logic is removed as it's now too complex to replicate the main.py setup here.
    logger.warning("Running bot.py directly is not recommended. Please run from main.py to ensure all dependencies are correctly injected.")
    logger.info("To run the application, execute: python -m RAG_BOT.src.main")
