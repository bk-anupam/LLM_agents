# /home/bk_anupam/code/LLM_agents/RAG_BOT/bot.py
import telebot
import sys
from telebot.types import Message, Update
from datetime import datetime
import re
import os
from flask import Flask, request, jsonify

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config import Config
from RAG_BOT.logger import logger
from vector_store import VectorStore
# Updated import for build_agent
from RAG_BOT.agent.graph_builder import build_agent
from langchain_core.messages import HumanMessage
from message_handler import MessageHandler
# AgentState might not be directly used here, but good practice if needed later
# from RAG_BOT.agent.state import AgentState


class TelegramBotApp:
    def __init__(self, config, vectordb, agent, handler):
        # Initialize Flask app
        self.app = Flask(__name__)
        self.config = config # Use injected config

        # Use injected dependencies
        self.vectordb = vectordb
        self.agent = agent
        self.handler = handler

        # Assumes a 'data' folder path exists in .env
        self.DATA_DIRECTORY = self.config.DATA_PATH
        logger.info(f"Data directory set to: {self.DATA_DIRECTORY}")

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

            # Register message handlers after bot initialization
            self.bot.register_message_handler(self.send_welcome, commands=['start'])
            self.bot.register_message_handler(self.send_help, commands=['help'])
            self.bot.register_message_handler(self.handle_document, content_types=['document'])
            self.bot.register_message_handler(self.handle_all_messages, func=lambda message: True)
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
                    self.bot.process_new_updates([update])
                    return jsonify({"status": "ok"})
                except Exception as e:
                    logger.error(f"Error processing webhook update: {e}", exc_info=True)
                    return jsonify({"status": "error", "message": "Internal server error"}), 500
            else:
                logger.warning(f"Received invalid content type for webhook: {request.headers.get('content-type')}")
                return jsonify({"status": "error", "message": "Invalid content type"}), 400

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
            self.handle_document,
            self.handle_all_messages,
        ]

    def send_welcome(self, message):
        logger.info(f"Received /start command from user {message.from_user.id}")
        self.bot.reply_to(message, "Welcome to the RAG Bot! Ask me questions about the indexed documents, or use /help for commands.")


    def send_help(self, message):
        logger.info(f"Received /help command from user {message.from_user.id}")
        self.bot.reply_to(message, 
            """
            Available Commands:
            /start - Show welcome message.
            /help - Show this help message.
            /query <your question> [date:YYYY-MM-DD] - Ask a question about the documents. Optionally filter by date.
            You can also just type your question directly.
            """
        )


    # --- Document Upload Handling (Consider if needed with startup indexing) ---    
    def handle_document(self, message: Message):
        """
        Handles incoming document messages. Checks for PDF, saves, and indexes.
        Note: Might re-index files if already processed at startup.
        """
        user_id = message.from_user.id
        if not message.document or not message.document.mime_type == 'application/pdf':
            logger.warning(f"User {user_id} uploaded non-PDF file: {message.document.mime_type if message.document else 'N/A'}")
            self.bot.reply_to(message, "Please upload a PDF document.")
            return

        file_name = message.document.file_name or f"doc_{message.document.file_id}.pdf"
        logger.info(f"User {user_id} uploaded PDF: {file_name} (file_id: {message.document.file_id})")

        try:
            file_info = self.bot.get_file(message.document.file_id)
            downloaded_file = self.bot.download_file(file_info.file_path)

            # Define a specific upload directory (maybe configurable)
            upload_dir = os.path.join(project_root, "uploads") # Example path
            os.makedirs(upload_dir, exist_ok=True)
            pdf_path = os.path.join(upload_dir, file_name)

            with open(pdf_path, 'wb') as new_file:
                new_file.write(downloaded_file)
            logger.info(f"PDF saved to: {pdf_path}")

            # Build the index using the VectorStore instance.
            # This will use the date check to potentially skip if already indexed.
            was_indexed = self.vector_store_instance.build_index(
                pdf_path,
                semantic_chunk=self.config.SEMANTIC_CHUNKING
            )

            if was_indexed:
                self.bot.reply_to(message, f"PDF '{file_name}' uploaded and indexed successfully.")
                # Optionally log metadata again after manual upload/index
                # self.vector_store_instance.log_all_indexed_metadata()
            else:
                # This message covers both skipped and failed cases from build_index
                 self.bot.reply_to(message, f"PDF '{file_name}' processed. It might have been skipped (already indexed) or encountered an issue during indexing. Check logs for details.")

        except Exception as e:
            logger.error(f"Error handling document upload from user {user_id}: {str(e)}", exc_info=True)
            self.bot.reply_to(message, "Sorry, I encountered an error processing your document.")
    # --- End Document Upload Handling ---


    def handle_all_messages(self, message: Message):
        """
        Handles all non-command text messages.
        """
        user_id = message.from_user.id
        logger.info(f"Received message from user {user_id}: '{message.text[:100]}...'")
        try:
            # Process the message using the handler (which might invoke the agent or query directly)
            response_text = self.handler.process_message(message)
            self.send_response(message, user_id, response_text)
        except Exception as e:
            logger.error(f"Error processing message from user {user_id}: {str(e)}", exc_info=True)
            self.bot.reply_to(message, "Sorry, I encountered an error processing your request.")


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

    def run(self):
        """Runs the Flask application."""
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
    try:
        # Initialize dependencies
        config = Config()

        # Assumes a 'data' folder path exists in .env
        DATA_DIRECTORY = config.DATA_PATH
        logger.info(f"Data directory set to: {DATA_DIRECTORY}")

        # Instantiate the vector store instance
        logger.info("Initializing VectorStore...")
        vector_store_instance = VectorStore(config.VECTOR_STORE_PATH)
        vectordb = vector_store_instance.get_vectordb() # Get the db instance after init
        logger.info("VectorStore initialized.")

        # --- Index data directory on startup ---        
        vector_store_instance.index_directory(DATA_DIRECTORY)        
        # --- End Indexing ---

        # Log the final state of indexed metadata after potential indexing
        logger.info("Logging final indexed metadata...")
        vector_store_instance.log_all_indexed_metadata()

        # Create rag agent instance
        logger.info("Initializing RAG agent...")
        # Ensure vectordb is valid before passing to agent
        if vectordb is None:
             logger.error("VectorDB instance is None after initialization and indexing. Cannot build agent.")
             exit(1)
        agent = build_agent(vectordb=vectordb, model_name=config.LLM_MODEL_NAME)
        logger.info("RAG agent initialized successfully")

        # Initialize message handler (for non-command messages)
        # Pass the agent instance to the handler
        handler = MessageHandler(agent=agent, config=config)

        # Create an instance of the TelegramBotApp and run it
        bot_app = TelegramBotApp(config=config, vectordb=vectordb, agent=agent, handler=handler)
        bot_app.run()

    except Exception as e:
        logger.critical(f"Failed during application startup: {str(e)}", exc_info=True)
        exit(1)

# Keep start_bot for potential polling mode if needed, but it's not used with webhook
# def start_bot():
#    logger.info("Starting bot in polling mode...")
#    bot.infinity_polling()
