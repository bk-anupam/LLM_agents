# /home/bk_anupam/code/LLM_agents/RAG_BOT/bot.py
import telebot
import asyncio
from telebot.types import Message, Update
import threading # Added for dedicated event loop
import os
from flask import Flask, request, jsonify
from RAG_BOT.config import Config
from RAG_BOT.logger import logger
from RAG_BOT.vector_store import VectorStore
from RAG_BOT.agent.graph_builder import build_agent
from langchain_core.messages import HumanMessage
from RAG_BOT.message_handler import MessageHandler
from RAG_BOT.utils import detect_document_language
from RAG_BOT.file_manager import FileManager 
from RAG_BOT.document_indexer import DocumentIndexer 
from RAG_BOT.pdf_processor import PdfProcessor
from RAG_BOT.htm_processor import HtmProcessor 
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from RAG_BOT.user_settings_manager import UserSettingsManager
from RAG_BOT.update_manager import UpdateManager
import sqlite3


class TelegramBotApp:
    def __init__(self, config: Config, vector_store_instance: VectorStore, 
                 pdf_processor: PdfProcessor = None, htm_processor: HtmProcessor = None):
        # Initialize Flask app
        self.app = Flask(__name__)
        self.config = config

        # Initialize the UserSettingsManager
        self.user_settings_manager = UserSettingsManager(db_path=config.SQLITE_DB_PATH)
        
        # Initialize the UpdateManager to handle duplicate messages
        self.update_manager = UpdateManager(db_path=config.SQLITE_DB_PATH)
        self.update_manager.cleanup_old_updates() # Clean up old records on startup

        # Initialize attributes that will be set later
        self.agent = None
        self.handler = None

        self.vector_store_instance = vector_store_instance
        self.pdf_processor = pdf_processor or PdfProcessor()
        self.htm_processor = htm_processor or HtmProcessor()         
        self._rag_bot_package_dir = os.path.abspath(os.path.dirname(__file__))
        self.project_root_dir = os.path.abspath(os.path.join(self._rag_bot_package_dir, '..'))

        # Vectordb can be initialized here
        self.vectordb = vector_store_instance.get_vectordb()

        # Assumes a 'data' folder path exists in .env
        self.DATA_DIRECTORY = self.config.DATA_PATH
        logger.info(f"Data directory set to: {self.DATA_DIRECTORY}")

        # Setup asyncio event loop for background tasks and agent invocations
        self.loop = asyncio.new_event_loop()
        # Make the thread non-daemon to prevent abrupt termination
        self.thread = threading.Thread(target=self._run_loop, daemon=False)
        self.thread.start()
        
        # Initialize Telegram bot and handlers
        self._initialize_telegram_bot()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        logger.info(f"Asyncio event loop starting in thread: {threading.current_thread().name}")
        try:
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"Asyncio event loop encountered an error: {e}", exc_info=True)
        finally:
            logger.info("Asyncio event loop is stopping...")
            self.loop.close()
            logger.info("Asyncio event loop has been closed.")

    async def initialize_agent_and_handler(self, vectordb, config: Config, checkpointer: BaseCheckpointSaver = None):
        """Initializes RAG agent and MessageHandler using the app's dedicated event loop."""
        logger.info("Initializing RAG agent and MessageHandler in dedicated loop...")
        # When this coroutine is run via run_coroutine_threadsafe on self.loop,
        # `await build_agent` will execute within self.loop's context.
        self.agent = await build_agent(vectordb=vectordb, config_instance=config, checkpointer=checkpointer)
        # MessageHandler itself is sync
        self.handler = MessageHandler(agent=self.agent, config=self.config) 
        logger.info("RAG agent and MessageHandler initialized successfully using dedicated loop.")

    def _initialize_telegram_bot(self):
        """Initializes the Telegram bot, webhook, and message handlers."""
        if not self.config.TELEGRAM_BOT_TOKEN:
            logger.error("TELEGRAM_BOT_TOKEN is not set. Please set it in your environment variables.")
            exit(1)
        try:
            # Create Telegram bot instance
            self.bot = telebot.TeleBot(self.config.TELEGRAM_BOT_TOKEN)
            logger.info("Telegram bot initialized successfully")
            # Setup webhook route after initializing bot and config
            self._setup_webhook_route()
            logger.info("Webhook route set up successfully")
            self._setup_health_check_route() # Add health check route
            logger.info("Health check route set up successfully")
            # Register message handlers after bot initialization
            self.bot.register_message_handler(self.send_welcome, commands=['start'])
            self.bot.register_message_handler(self.send_help, commands=['help'])
            self.bot.register_message_handler(self.handle_language_command, commands=['language'])
            self.bot.register_message_handler(self.handle_mode_command, commands=['mode']) # Register new command for mode
            self.bot.register_message_handler(self.handle_document, content_types=['document'])
            # Handle all other messages that are not commands or documents
            self.bot.register_message_handler(self.handle_all_messages_wrapper, func=lambda message: True, content_types=['text'])
            logger.info("Message handlers registered successfully")
        except Exception as e:
            logger.critical(f"Failed during application startup: {str(e)}", exc_info=True)
            exit(1)


    def _setup_webhook_route(self):
        """Sets up the webhook endpoint for Telegram."""
        @self.app.route(f'/{self.config.TELEGRAM_BOT_TOKEN}', methods=['POST'])
        def webhook():
            """Handle incoming webhook requests from Telegram"""
            if request.headers.get('content-type') == 'application/json':
                logger.info("Received webhook request") # Changed level to debug for less noise
                try:
                    json_data = request.get_json()
                    update = Update.de_json(json_data)
                    
                    # Idempotency check to prevent processing duplicate updates during cold starts
                    if self.update_manager.is_update_processed(update.update_id):
                        logger.info(f"Duplicate update ID {update.update_id} received, ignoring.")
                        return jsonify({"status": "ok, duplicate"})
                    
                    self.update_manager.mark_update_as_processed(update.update_id)

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
        

    def send_response(self, message, user_id, response_text):
        """
        Sends a response to the user, handling potential message length limits.
        """
        if not response_text:
            logger.warning(f"Attempted to send empty response to user {user_id}")
            response_text = "Sorry, I could not generate a response."

        # Maximum allowed message length in Telegram (adjust if needed)
        max_telegram_length = 4096
        chunks = [response_text[i:i + max_telegram_length] for i in range(0, len(response_text), max_telegram_length)]
        try:
            # Send first chunk as reply, subsequent as regular messages to the chat
            if chunks:
                logger.info(f"Sending response to user {user_id}: {chunks[0][:100]}...")
                self.bot.reply_to(message, chunks[0])
                for chunk in chunks[1:]:
                    self.bot.send_message(message.chat.id, chunk)
        except telebot.apihelper.ApiException as e:
            logger.error(f"Error sending message chunk to user {user_id} in chat {message.chat.id}: {str(e)}")
            # Maybe try sending a generic error message if the main response failed
            try:
                self.bot.reply_to(message, "Sorry, there was an error sending the full response.")
            except Exception:
                logger.error(f"Failed even to send error notification to user {user_id}")
        except Exception as e:
             logger.error(f"Unexpected error in send_response for user {user_id}: {e}", exc_info=True)


    # Telegram message handlers
    @property
    def message_handlers(self):
        """Returns a list of message handlers for the bot."""
        return [
            self.send_welcome,
            self.send_help,
            self.handle_language_command,
            self.handle_document,
            self.handle_all_messages_wrapper,
        ]


    def send_welcome(self, message):
        logger.info(f"Received /start command from user {message.from_user.id}")
        self.bot.reply_to(message, "Welcome to the spiritual chatbot! Ask me questions about the indexed documents, or use /help for commands.")


    def send_help(self, message):
        logger.info(f"Received /help command from user {message.from_user.id}")
        self.bot.reply_to(message,
            """
            Available Commands:
            /start - Show welcome message.
            /help - Show this help message.
            /language <lang> - Set bot language (english or hindi). Example: /language hindi
            /mode <mode> - Set bot response mode (default or research). Example: /mode research
            /query <your question> [date:YYYY-MM-DD] - Ask a question about the documents. Optionally filter by date.
            You can also just type your question directly.
            """
        )


    def handle_language_command(self, message: Message):
        """Handles the /language command to set user preference."""
        user_id = message.from_user.id
        parts = message.text.split(maxsplit=1)

        if len(parts) < 2:
            usage_text = self.config.get_user_message('language_usage_help', "Usage: /language <english|hindi>")
            self.bot.reply_to(message, usage_text)
            return

        lang_input = parts[1].strip().lower()
        lang_code = {'english': 'en', 'hindi': 'hi'}.get(lang_input)

        if not lang_code:
            unsupported_text = self.config.get_user_message('language_unsupported', "Unsupported language. Please use 'english' or 'hindi'.")
            self.bot.reply_to(message, unsupported_text)
            return

        # Submit the async task to the dedicated event loop
        future = asyncio.run_coroutine_threadsafe(
            self.user_settings_manager.update_user_settings(user_id, language_code=lang_code),
            self.loop
        )
        try:
            future.result(timeout=10) # Short timeout for DB update
            logger.info(f"Successfully set language for user {user_id} to '{lang_code}'")
            confirmation_key = f"language_set_{lang_code}"
            defaults = {'en': "Language set to English.", 'hi': "भाषा हिंदी में सेट कर दी गई है।"}
            reply_text = self.config.get_user_message(confirmation_key, defaults[lang_code])
            self.bot.reply_to(message, reply_text)
        except Exception as e:
            logger.error(f"Failed to update language settings for user {user_id}: {e}", exc_info=True)
            self.bot.reply_to(message, "Sorry, there was an error saving your preference.")


    def handle_mode_command(self, message: Message):
        """Handles the /mode command to set user response mode."""
        user_id = message.from_user.id
        parts = message.text.split(maxsplit=1)

        async def get_and_reply_current_mode():
            settings = await self.user_settings_manager.get_user_settings(user_id)
            self.bot.reply_to(message, f"Current mode is '{settings['mode']}'. Usage: /mode <default|research>")

        if len(parts) < 2:
            asyncio.run_coroutine_threadsafe(get_and_reply_current_mode(), self.loop)
            return

        new_mode = parts[1].strip().lower()
        if new_mode not in ['default', 'research']:
            self.bot.reply_to(message, "Invalid mode. Please use 'default' or 'research'.")
            return

        future = asyncio.run_coroutine_threadsafe(
            self.user_settings_manager.update_user_settings(user_id, mode=new_mode),
            self.loop
        )
        try:
            future.result(timeout=10)
            logger.info(f"Set mode preference for user {user_id} to '{new_mode}'")
            self.bot.reply_to(message, f"Mode set to '{new_mode}'.")
        except Exception as e:
            logger.error(f"Failed to update mode settings for user {user_id}: {e}", exc_info=True)
            self.bot.reply_to(message, "Sorry, there was an error saving your preference.")


    def _cleanup_uploaded_file(self, file_path, processed_successfully):
        """Handles cleanup of uploaded files after processing."""
        if processed_successfully and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Successfully processed and removed '{file_path}' from uploads directory.")
            except OSError as e:
                logger.error(f"Error removing processed file '{file_path}' from uploads: {e}")
        elif not processed_successfully and os.path.exists(file_path):
            logger.info(f"File '{file_path}' was not successfully processed/indexed and will remain in the uploads directory.")
        elif not os.path.exists(file_path) and processed_successfully:
            logger.warning(f"Attempted to remove '{file_path}', but it was already deleted (or never saved properly).")

    def _determine_file_name(self, message, file_ext, default_doc_name):
        """Determines the correct file name for the uploaded document."""
        original_file_name = message.document.file_name
        file_name = original_file_name or default_doc_name
        # Ensure the filename has the correct extension if it was defaulted
        if not file_name.lower().endswith(file_ext) and original_file_name is None:
            file_name = os.path.splitext(file_name)[0] + file_ext
        return file_name

    def _process_document_metadata(self, message: Message):
        """
        Determines file extension, default name, and processing mime type
        based on the uploaded document's mime type and filename.
        Returns a tuple: (file_ext, default_doc_name, processing_mime_type)
        Raises ValueError if the file type is unsupported.
        """
        mime_type = message.document.mime_type
        file_id = message.document.file_id
        original_file_name = message.document.file_name

        file_ext = None
        processing_mime_type = mime_type # Default to original mime type

        if mime_type == 'application/pdf':
            file_ext = ".pdf"
            default_doc_name = f"doc_{file_id}.pdf"
        elif mime_type in ['text/html', 'application/xhtml+xml']:
            file_ext = ".htm"
            default_doc_name = f"doc_{file_id}.htm"
        elif mime_type == 'application/octet-stream':
             # If generic binary, try to determine type from file name
             if original_file_name:
                 name, ext = os.path.splitext(original_file_name)
                 if ext.lower() in ['.htm', '.html']:
                     file_ext = ".htm"
                     default_doc_name = original_file_name
                     processing_mime_type = 'text/html' 
                 elif ext.lower() == '.pdf':
                     file_ext = ".pdf"
                     default_doc_name = original_file_name
                     processing_mime_type = 'application/pdf' 
             
             if file_ext is None: # If still no specific type determined
                 raise ValueError(f"Unsupported file type or unable to determine type from '{original_file_name or 'uploaded file'}'.")

        else: # Handle other explicit unsupported mime types
            raise ValueError(f"Unsupported file type ({mime_type}).")

        return file_ext, default_doc_name, processing_mime_type


    # --- Document Upload Handling (Consider if needed with startup indexing) ---
    def handle_document(self, message: Message):
        """
        Handles incoming document messages. Checks for PDF, saves, and indexes.
        Detects language using utility function.
        """
        user_id = message.from_user.id
        if not message.document:
            self.bot.reply_to(message, "No document provided.")
            return

        file_id = message.document.file_id
        mime_type = message.document.mime_type # Keep original mime_type for logging initially
        logger.info(f"Received document from user mime_type: {mime_type} (file_id: {file_id})")
        file_path = None # Initialize file_path
        documents = [] # Initialize documents list
        processed_successfully = False
        try:
            # Use the new helper method to process metadata
            file_ext, default_doc_name, processing_mime_type = self._process_document_metadata(message)
            file_name = self._determine_file_name(message, file_ext, default_doc_name)
            logger.info(f"User {user_id} uploaded {mime_type} (processed as {processing_mime_type}): {file_name}")
            # Define a specific upload directory
            upload_dir = os.path.join(self.project_root_dir , "uploads")
            logger.debug(f"Upload directory: {upload_dir}")
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, file_name)                    
            file_info = self.bot.get_file(file_id)
            downloaded_file = self.bot.download_file(file_info.file_path)
            with open(file_path, 'wb') as new_file:
                new_file.write(downloaded_file)
            logger.info(f"Document saved to: {file_path}")
            # Load the document using the appropriate processor based on processing_mime_type
            if processing_mime_type == 'application/pdf':
                documents = self.pdf_processor.load_pdf(file_path)
            elif processing_mime_type in ['text/html', 'application/xhtml+xml']:
                # HtmProcessor.load_htm returns a single Document or None
                doc = self.htm_processor.load_htm(file_path)
                if doc:
                    documents.append(doc)
            
            if not documents:
                logger.warning(f"No documents loaded from: {file_path}. Skipping indexing.")
                self.bot.reply_to(message, f"Could not load content from '{file_name}'.")
                # File remains in uploads dir if loading fails
                return

            # Detect language using the utility function with loaded documents
            language = detect_document_language(documents, file_name_for_logging=file_name)
            if language not in ['en', 'hi']:
                logger.warning(f"Unsupported language detected: {language}. Aborting document indexing.")
                self.bot.reply_to(message, f"Unsupported language '{language}' detected in '{file_name}'. Indexing aborted.")
                return 
            # Add detected language metadata
            for doc in documents:
                doc.metadata['language'] = language
            logger.info(f"Added language metadata '{language}' to uploaded document: {file_name}")
            # Index the document list
            was_indexed = self.vector_store_instance.index_document(documents, semantic_chunk=self.config.SEMANTIC_CHUNKING)                        
            if was_indexed:
                self.bot.reply_to(message, f"Document '{file_name}' uploaded and indexed successfully.")
                processed_successfully = True
            else:
                self.bot.reply_to(message, f"Document '{file_name}' was not indexed (possibly already exists or an error occurred).")
                # File remains in uploads dir if indexing fails or it's a duplicate

        except ValueError as ve: # Catch unsupported file type errors from _process_document_metadata
             logger.warning(f"Unsupported file type for user {user_id}: {ve}")
             self.bot.reply_to(message, str(ve))
             # No file was saved in this case, so no cleanup needed
             return
        except Exception as e:
            logger.error(f"Error handling document upload from user {user_id} for {file_name}: {str(e)}", exc_info=True)
            self.bot.reply_to(message, "Sorry, I encountered an error processing your document.")            
        finally:
            # Delete the file from upload_dir ONLY if processed and indexed successfully
            # Ensure file_path is not None before attempting cleanup
            if file_path:
                self._cleanup_uploaded_file(file_path, processed_successfully)

    # --- End Document Upload Handling ---

    def handle_all_messages_wrapper(self, message: Message):
        """Synchronous wrapper for the async message handler."""
        user_id = message.from_user.id
        if not self.handler:
            logger.error(f"MessageHandler not initialized. Cannot process message for user {user_id}.")
            self.send_response(message, user_id, "The bot is currently initializing. Please try again shortly.")
            return

        response_text = "Sorry, an unexpected error occurred." # Default response
        try:
            # Submit the async task to the dedicated event loop
            future = asyncio.run_coroutine_threadsafe(self._handle_all_messages_async_core(message), self.loop)                        
            response_text = future.result(timeout=self.config.ASYNC_OPERATION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"Async message processing timed out for user {user_id}")
            response_text = "Sorry, your request timed out. Please try again."
        except Exception as e:
            logger.error(f"Error during async message processing for user {user_id}: {e}", exc_info=True)
            # response_text remains the default "Sorry, an unexpected error occurred."
        finally:
            # Always send a response back to the user.
            self.send_response(message, user_id, response_text)


    async def _handle_all_messages_async_core(self, message: Message) -> str:
        """
        Core async logic for handling messages. Runs in the dedicated event loop.
        Returns the response string.
        """
        user_id = message.from_user.id
        
        # Get user settings from the database
        settings = await self.user_settings_manager.get_user_settings(user_id)
        user_lang = settings.get('language_code', 'en')
        user_mode = settings.get('mode', 'default')
        
        logger.info(f"Processing message for user {user_id} in async core: '{message.text[:100]}...' (Lang: {user_lang}, Mode: {user_mode})")

        try:
            # Pass the mode to the message handler
            return await self.handler.process_message(message, user_lang, user_mode)
        except Exception as e:
            logger.error(f"Error in _handle_all_messages_async_core for user {user_id}: {str(e)}", exc_info=True)
            return "Sorry, I encountered an internal error while processing your request."

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
        if not self.agent or not self.handler:
            logger.critical("Agent or MessageHandler not initialized before run(). Ensure initialize_agent_and_handler() was successfully called.")
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


if __name__ == "__main__":
    # This top-level function is now synchronous as it orchestrates
    # the creation of the app and then runs its async initialization.
    def main_setup_and_run():
        try:
            config = Config()
            # Initialize in-memory checkpointer for async compatibility
            checkpointer = InMemorySaver()
            logger.info("InMemorySaver checkpointer initialized successfully.")
            
            # Assumes a 'data' folder path exists in .env
            DATA_DIRECTORY = config.DATA_PATH
            logger.info(f"Data directory set to: {DATA_DIRECTORY}")
            
            vector_store_instance = VectorStore(persist_directory=config.VECTOR_STORE_PATH, config=config)
            vectordb = vector_store_instance.get_vectordb() # Get the db instance after init
            logger.info("VectorStore initialized.")
            # --- Index data directory on startup ---            
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

            if vectordb is None:
                logger.error("VectorDB instance is None. Cannot proceed.")
                exit(1)

            pdf_processor = PdfProcessor() 
            htm_processor = HtmProcessor() 

            # Create bot_app instance. This starts its event loop thread.
            bot_app = TelegramBotApp(config=config, vector_store_instance=vector_store_instance,
                                        pdf_processor=pdf_processor, htm_processor=htm_processor)

            # Define an async function to perform async initializations
            async def _async_init_for_bot_app():
                await bot_app.initialize_agent_and_handler(vectordb, config, checkpointer=checkpointer)

            # Run the async initialization (agent building, handler creation) on the bot_app's dedicated event loop.
            logger.info("Submitting async initialization to bot's event loop...")
            future = asyncio.run_coroutine_threadsafe(_async_init_for_bot_app(), bot_app.loop)
            try:
                future.result(timeout=config.ASYNC_OPERATION_TIMEOUT) 
                logger.info("Async initialization of agent and handler complete.")
            except Exception as e:
                logger.critical(f"Failed to initialize agent and handler via dedicated loop: {e}", exc_info=True)
                bot_app.loop.call_soon_threadsafe(bot_app.loop.stop) # Request loop to stop
                bot_app.thread.join(timeout=5) # Wait for thread to finish
                exit(1)

            # At this point, bot_app.agent and bot_app.handler are set.
            # Now, run the Flask app.
            bot_app.run()

        except Exception as e:
            logger.critical(f"Failed during application startup: {str(e)}", exc_info=True)
            exit(1)

    main_setup_and_run()