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
from rag_agent import build_agent
from langchain_core.messages import HumanMessage
from message_handler import MessageHandler


# Initialize Flask app
app = Flask(__name__)
config = Config()

# --- Define the data directory ---
# Assumes a 'data' folder path exists in .env 
DATA_DIRECTORY = config.DATA_PATH
logger.info(f"Data directory set to: {DATA_DIRECTORY}")
# --- End Data Directory Definition ---


if not config.TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN is not set. Please set it in your environment variables.")
    exit(1)  # Exit the script if token is missing

try:
    # Create Telegram bot instance
    bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
    logger.info("Telegram bot initialized successfully")

    # Instantiate the vector store instance
    logger.info("Initializing VectorStore...")
    vector_store_instance = VectorStore(config.VECTOR_STORE_PATH)
    vectordb = vector_store_instance.get_vectordb() # Get the db instance after init
    logger.info("VectorStore initialized.")

    # --- Index data directory on startup ---
    logger.info(f"Starting indexing of PDF documents in: {DATA_DIRECTORY}")
    vector_store_instance.index_directory(DATA_DIRECTORY)
    logger.info("Finished indexing data directory.")
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

except Exception as e:
    # Catch initialization errors more broadly
    logger.critical(f"Failed during application startup: {str(e)}", exc_info=True)
    exit(1)


# Webhook endpoint for Telegram
@app.route(f'/{config.TELEGRAM_BOT_TOKEN}', methods=['POST'])
def webhook():
    """Handle incoming webhook requests from Telegram"""
    if request.headers.get('content-type') == 'application/json':
        logger.info("Received webhook request") # Changed level to debug for less noise
        try:
            json_data = request.get_json()
            update = Update.de_json(json_data)
            bot.process_new_updates([update])
            return jsonify({"status": "ok"})
        except Exception as e:
            logger.error(f"Error processing webhook update: {e}", exc_info=True)
            return jsonify({"status": "error", "message": "Internal server error"}), 500
    else:
        logger.warning(f"Received invalid content type for webhook: {request.headers.get('content-type')}")
        return jsonify({"status": "error", "message": "Invalid content type"}), 400


def send_response(message, user_id, response_text):
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
            bot.reply_to(message, chunks[0])
            for chunk in chunks[1:]:
                bot.send_message(message.chat.id, chunk)
    except telebot.apihelper.ApiException as e:
        logger.error(f"Error sending message chunk to user {user_id} in chat {message.chat.id}: {str(e)}")
        # Maybe try sending a generic error message if the main response failed
        try:
            bot.reply_to(message, "Sorry, there was an error sending the full response.")
        except Exception:
            logger.error(f"Failed even to send error notification to user {user_id}")
    except Exception as e:
         logger.error(f"Unexpected error in send_response for user {user_id}: {e}", exc_info=True)


# Telegram message handlers
@bot.message_handler(commands=['start'])
def send_welcome(message):
    logger.info(f"Received /start command from user {message.from_user.id}")
    bot.reply_to(message, "Welcome to the RAG Bot! Ask me questions about the indexed documents, or use /help for commands.")


@bot.message_handler(commands=['help'])
def send_help(message):
    logger.info(f"Received /help command from user {message.from_user.id}")
    bot.reply_to(message, """
    Available Commands:
    /start - Show welcome message.
    /help - Show this help message.
    /query <your question> [date:YYYY-MM-DD] - Ask a question about the documents. Optionally filter by date.
    You can also just type your question directly.
    """)


# --- Document Upload Handling (Consider if needed with startup indexing) ---
# If you index everything on startup, you might disable or modify this handler.
# Keeping it for now, but be aware it might re-index files already processed at startup.
@bot.message_handler(content_types=['document'])
def handle_document(message: Message):
    """
    Handles incoming document messages. Checks for PDF, saves, and indexes.
    Note: Might re-index files if already processed at startup.
    """
    user_id = message.from_user.id
    if not message.document or not message.document.mime_type == 'application/pdf':
        logger.warning(f"User {user_id} uploaded non-PDF file: {message.document.mime_type if message.document else 'N/A'}")
        bot.reply_to(message, "Please upload a PDF document.")
        return

    file_name = message.document.file_name or f"doc_{message.document.file_id}.pdf"
    logger.info(f"User {user_id} uploaded PDF: {file_name} (file_id: {message.document.file_id})")

    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Define a specific upload directory (maybe configurable)
        upload_dir = os.path.join(project_root, "uploads") # Example path
        os.makedirs(upload_dir, exist_ok=True)
        pdf_path = os.path.join(upload_dir, file_name)

        with open(pdf_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        logger.info(f"PDF saved to: {pdf_path}")

        # Build the index using the VectorStore instance.
        # This will use the date check to potentially skip if already indexed.
        was_indexed = vector_store_instance.build_index(
            pdf_path,
            semantic_chunk=config.SEMANTIC_CHUNKING
        )

        if was_indexed:
            bot.reply_to(message, f"PDF '{file_name}' uploaded and indexed successfully.")
            # Optionally log metadata again after manual upload/index
            # vector_store_instance.log_all_indexed_metadata()
        else:
            # This message covers both skipped and failed cases from build_index
             bot.reply_to(message, f"PDF '{file_name}' processed. It might have been skipped (already indexed) or encountered an issue during indexing. Check logs for details.")

    except Exception as e:
        logger.error(f"Error handling document upload from user {user_id}: {str(e)}", exc_info=True)
        bot.reply_to(message, "Sorry, I encountered an error processing your document.")
# --- End Document Upload Handling ---


@bot.message_handler(commands=['query'])
def handle_query(message: Message):
    """
    Handles the /query command, extracts query and optional date filter.
    """
    user_id = message.from_user.id
    logger.info(f"Received /query command from user {user_id}")
    try:
        # Extract query and optional parameters
        full_command = message.text[len('/query '):].strip()
        date_match = re.search(r'\sdate:(\d{4}-\d{2}-\d{2})\s*$', full_command) # Match date at the end
        date_filter = None
        query = full_command
        if date_match:
            date_filter = date_match.group(1)
            # Remove the date part from the query string
            query = full_command[:date_match.start()].strip()
            logger.info(f"Extracted date filter: {date_filter} for user {user_id}")

        if not query:
            logger.warning(f"Empty query received from user {user_id} with /query command.")
            bot.reply_to(message, "Please provide a question after the /query command.\nUsage: `/query your question here [date:YYYY-MM-DD]`")
            return

        logger.info(f"Processing query from user {user_id}: '{query[:50]}...'")

        # Use the VectorStore's query method directly
        response_text = vector_store_instance.query_index(
            query=query,
            k=config.RETRIEVER_K, # Use config for K value
            model_name=config.LLM_MODEL_NAME,
            date_filter=date_filter
        )

        send_response(message, user_id, response_text)

    except Exception as e:
        logger.error(f"Error handling /query from user {user_id}: {str(e)}", exc_info=True)
        bot.reply_to(message, "Sorry, I encountered an error processing your query.")


# Initialize message handler (for non-command messages)
# Consider if the RAG agent logic should be directly in the message handler
# or if the VectorStore query method is sufficient for simple Q&A.
# Assuming MessageHandler uses the agent for more complex interactions if needed.
handler = MessageHandler(agent=agent, config=config)


@bot.message_handler(func=lambda message: True)
def handle_all_messages(message: Message):
    """
    Handles all non-command text messages.
    """
    user_id = message.from_user.id
    logger.info(f"Received message from user {user_id}: '{message.text[:100]}...'")
    try:
        # Process the message using the handler (which might invoke the agent or query directly)
        response_text = handler.process_message(message)
        send_response(message, user_id, response_text)
    except Exception as e:
        logger.error(f"Error processing message from user {user_id}: {str(e)}", exc_info=True)
        bot.reply_to(message, "Sorry, I encountered an error processing your request.")


# Setup and webhook configuration functions
def setup_webhook(url):
    """Set up the webhook for the Telegram bot"""
    if not url:
        logger.error("WEBHOOK_URL is not configured. Cannot set webhook.")
        return False # Indicate failure
    try:
        webhook_url = f"{url.rstrip('/')}/{config.TELEGRAM_BOT_TOKEN}"
        logger.info("Removing existing webhook (if any)...")
        bot.remove_webhook()
        logger.info(f"Setting webhook to: {webhook_url}")
        success = bot.set_webhook(url=webhook_url)
        if success:
            logger.info("Webhook set successfully.")
            return True
        else:
            logger.error("Failed to set webhook.")
            return False
    except Exception as e:
        logger.error(f"Error setting up webhook: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Set the webhook URL before starting the Flask app
    WEBHOOK_URL = config.WEBHOOK_URL
    if not WEBHOOK_URL:
        logger.error("WEBHOOK_URL is not set in config. Cannot start Flask server with webhook.")
        # Decide if you want to exit or maybe run in polling mode as a fallback
        # For now, let's exit if webhook URL is needed but not provided
        # Consider adding a command-line flag to choose polling vs webhook
        exit(1)

    if setup_webhook(WEBHOOK_URL):
        # Start the Flask app only if webhook setup was successful
        logger.info(f"Starting Flask server on port {config.PORT}")
        # Use waitress or gunicorn for production instead of Flask's development server
        app.run(host='0.0.0.0', port=config.PORT, debug=False)
    else:
        logger.critical("Failed to set up webhook. Aborting Flask server start.")
        exit(1)

# Keep start_bot for potential polling mode if needed, but it's not used with webhook
# def start_bot():
#    logger.info("Starting bot in polling mode...")
#    bot.infinity_polling()
