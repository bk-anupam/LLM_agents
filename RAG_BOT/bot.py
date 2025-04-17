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
from logger import logger
from vector_store import VectorStore
from rag_agent import build_agent
from langchain_core.messages import HumanMessage
from message_handler import MessageHandler


# Initialize Flask app
app = Flask(__name__)
config = Config()

if not config.TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN is not set. Please set it in your environment variables.")
    exit(1)  # Exit the script if token is missing

try:
    # Create Telegram bot instance
    bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
    logger.info("Telegram bot initialized successfully")
     # Instantiate and get the vector store instance
    vector_store_instance = VectorStore(config.VECTOR_STORE_PATH)
    vectordb = vector_store_instance.get_vectordb()
    logger.info("Got vectordb instance from VectorStore class.")
    # create rag agent instance    
    agent = build_agent(vectordb=vectordb, model_name=config.LLM_MODEL_NAME)
    logger.info("RAG agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Telegram bot: {str(e)}")
    exit(1)


# Webhook endpoint for Telegram
@app.route(f'/{config.TELEGRAM_BOT_TOKEN}', methods=['POST'])
def webhook():
    """Handle incoming webhook requests from Telegram"""
    if request.headers.get('content-type') == 'application/json':
        logger.info("Received webhook request")
        json_data = request.get_json()
        update = Update.de_json(json_data)
        bot.process_new_updates([update])
        return jsonify({"status": "ok"})
    else:
        return jsonify({"status": "error", "message": "Invalid content type"}), 400


def send_response(message, user_id, response_text):
    """
    Sends a response to the user, handling potential message length limits.
    Args:
        message (telebot.types.Message): The original Telegram message object.  Needed for `bot.reply_to`.
        user_id (int): The Telegram user ID. Used for logging.
        response_text (str): The text of the response to send.
    Returns:
        None
    """
    # Maximum allowed message length in Telegram (adjust if needed)
    max_telegram_length = 4096  
    chunks = [response_text[i:i + max_telegram_length] for i in range(0, len(response_text), max_telegram_length)]
    logger.info(f"Sending query response to user {user_id}")  
    for chunk in chunks:
        try:
            # Send each chunk as a separate message
            bot.reply_to(message, chunk)  
        except telebot.apihelper.ApiException as e:
            logger.error(f"Error sending message chunk to user {user_id}: {str(e)}")  


# Telegram message handlers
@bot.message_handler(commands=['start'])
def send_welcome(message):
    print("Received /start command")
    bot.reply_to(message, "Welcome to the Telegram Bot! Type a message to get started or send /help for available commands.")


@bot.message_handler(commands=['help'])
def send_help(message):
    print("Received /help command")
    bot.reply_to(message, """
        Here are the available commands:
        /start - Start the bot
        /help - Show this help message
        /query - Query the bot with a message related to the uploaded PDF documents
            
        You can also just type a message and I'll respond based on my programming!
        """)


@bot.message_handler(content_types=['document'])
def handle_document(message: Message):
    """
    Handles incoming document messages. This function checks if the uploaded document is a PDF,
    downloads it, saves it to the local filesystem, and indexes it for later retrieval.
    Args:
        message (Message): The incoming message containing the document.
    Returns:
        None
    """
    if not message.document.mime_type == 'application/pdf':
        bot.reply_to(message, "Please upload a PDF document.")
        return
    try:
        logger.info(f"Downloading file with file_id:{message.document.file_id}")
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        os.makedirs("uploads", exist_ok=True)
        pdf_path = os.path.join("uploads", message.document.file_name)
        with open(pdf_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)
        # Build the index using the VectorStore instance.
        vector_store_instance.build_index(pdf_path, semantic_chunk=config.SEMANTIC_CHUNKING)
        bot.reply_to(message, "PDF uploaded and indexed successfully.")
    except Exception as e:
        logger.error(f"Error handling document: {str(e)}")
        bot.reply_to(message, "Sorry, I encountered an error processing your document.")


@bot.message_handler(commands=['query'])
def handle_query(message: Message):
    """
    Handles the /query command from a Telegram message.
    Supports an optional date filter in the format: /query <your_query> date:YYYY-MM-DD
    """
    try:
        # Extract query and optional parameters
        full_query = message.text[len('/query '):].strip()
        date_match = re.search(r'date:(\d{4}-\d{2}-\d{2})', full_query)
        date_filter = date_match.group(1) if date_match else None
        query = re.sub(r'date:\d{4}-\d{2}-\d{2}', '', full_query).strip()
        if not query:
            bot.reply_to(message, "Please provide a query after the /query command.")
            return

        # Build the initial state for the agent
        user_message = query
        if date_filter:
            user_message += f" (Date: {date_filter})"
        initial_state = {"messages": [HumanMessage(content=user_message)]}

        # Invoke the agent
        final_state = agent.invoke(initial_state)
        answer = None
        if isinstance(final_state, dict) and "messages" in final_state and final_state["messages"]:
            last_msg = final_state["messages"][-1]
            if hasattr(last_msg, "content"):
                answer = last_msg.content
        if not answer:
            answer = "Sorry, I couldn't find an answer."
        send_response(message, message.from_user.id, answer)
    except Exception as e:
        logger.error(f"Error handling query: {str(e)}")
        bot.reply_to(message, "Sorry, I encountered an error processing your query.")


# Initialize message handler
handler = MessageHandler(agent=agent, config=config)


@bot.message_handler(func=lambda message: True)
def handle_all_messages(message: Message):
    """
    Handles all incoming messages to the bot.
    This function is triggered for every message received by the bot. It logs the 
    message details, processes the message using the handler, and sends a response 
    back to the user. If an error occurs during message processing, it logs the 
    error and sends an error message to the user.
    Args:
        message (telebot.types.Message): The incoming message object containing 
        details about the message and the user.
    Returns:
        None
    """
    try:        
        # Process the message
        response_text = handler.process_message(message)                
        send_response(message, message.from_user.id, response_text)                
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        bot.reply_to(message, "Sorry, I encountered an error processing your request.")


# Setup and webhook configuration functions
def setup_webhook(url):
    """Set up the webhook for the Telegram bot"""
    webhook_url = f"{url}/{config.TELEGRAM_BOT_TOKEN}"
    bot.remove_webhook()
    bot.set_webhook(url=webhook_url)
    logger.info(f"Webhook set to: {webhook_url}")
    return webhook_url

if __name__ == "__main__":
    # Set the webhook URL before starting the Flask app
    # Replace with your actual public URL where this Flask app will be hosted
    WEBHOOK_URL = config.WEBHOOK_URL  # This should be set in your Config class
    setup_webhook(WEBHOOK_URL)
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=config.PORT, debug=False)


def start_bot():
    bot.infinity_polling()
