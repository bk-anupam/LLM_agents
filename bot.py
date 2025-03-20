import telebot
from telebot.types import Message, Update
from config import Config
from logger import logger
from datetime import datetime
import re
import os
from vector_store import build_index, load_existing_index, query_index, query_llm
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)
config = Config()

# Create Telegram bot instance
if not config.TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN is not set. Please set it in your environment variables.")
    exit(1)  # Exit the script if token is missing

# Create Telegram bot instance
try:
    bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
    logger.info("Telegram bot initialized successfully")
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
    logger.info(f"Sending query response to user {user_id}: {response_text}")  
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
    # Check if the uploaded document is a PDF
    if not message.document.mime_type == 'application/pdf':
        bot.reply_to(message, "Please upload a PDF document.")
        return
    try:
        # Get file information and download the file
        logger.info(f"Downloading file with file_id:{message.document.file_id}")
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        # Ensure the uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        pdf_path = os.path.join("uploads", message.document.file_name)        
        # Save the downloaded file to the uploads directory
        with open(pdf_path, 'wb') as new_file:
            new_file.write(downloaded_file)        
        # Ensure the chroma_db directory exists
        os.makedirs("chroma_db", exist_ok=True)
        # Build the index for the uploaded PDF
        build_index(pdf_path, persist_directory=config.VECTOR_STORE_PATH, semantic_chunk=True)
        bot.reply_to(message, "PDF uploaded and indexed successfully.")
    except Exception as e:
        logger.error(f"Error handling document: {str(e)}")
        bot.reply_to(message, "Sorry, I encountered an error processing your document.")


@bot.message_handler(commands=['query'])
def handle_query(message):
    """
    Handles the /query command from a Telegram message.
    Supports an optional date filter in the format: /query <your_query> date:YYYY-MM-DD
    """
    try:
        # Extract query and optional date filter
        full_query = message.text[len('/query '):].strip()
        date_match = re.search(r'date:(\d{4}-\d{2}-\d{2})', full_query)
        date_filter = date_match.group(1) if date_match else None
        query = re.sub(r'date:\d{4}-\d{2}-\d{2}', '', full_query).strip()

        if not query:
            bot.reply_to(message, "Please provide a query after the /query command.")
            return

        logger.info(f"Received query: {query}, Date filter: {date_filter}")
        vectordb = load_existing_index(persist_directory=config.VECTOR_STORE_PATH)
        result = query_index(vectordb, query, date_filter=date_filter)  # Pass the date_filter        
        send_response(message, message.from_user.id, result)
    except Exception as e:
        logger.error(f"Error handling query: {str(e)}")
        bot.reply_to(message, "Sorry, I encountered an error processing your query.")


# Message handler class
class MessageHandler:
    def __init__(self):
        # Store user session data (could be moved to a database for persistence)
        self.sessions = config.USER_SESSIONS
    
    def _get_user_session(self, user_id):
        """Get or create a new session for the user"""
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                'last_interaction': datetime.now(),
                'conversation': [],
                'context': {}
            }
        return self.sessions[user_id]
    
    def _update_session(self, user_id, message, response):
        """Update the user session with new interaction"""
        session = self._get_user_session(user_id)
        session['last_interaction'] = datetime.now()
        session['conversation'].append({
            'user': message,
            'bot': response,
            'timestamp': datetime.now().isoformat()
        })
        # Limit conversation history (optional)
        if len(session['conversation']) > 10:
            session['conversation'] = session['conversation'][-10:]
    
    def process_message(self, user_id, message):
        """
        Process the incoming message and generate a response
        This is where you implement your custom logic
        """
        # Get user session
        session = self._get_user_session(user_id)
        
        # Convert message to lowercase for easier matching
        message_lower = message.lower().strip()
        
        # Basic response logic
        if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey']):
            response = "👋 Hello! I'm your Telegram assistant. How can I help you today?"
        
        elif "help" in message_lower:
            response = ("Here's what I can do:\n"
                      "- Answer your query (on pdf documents uploaded). Use /query command followed by the query for this\n"
                      "- index and store in vector DB uploaded pdf documents. Just send the pdf document as a message\n"
                      "- answer any general query \n" 
                      "- last message - to see your last message\n"                     
                      "Just let me know what you need!")                
        
        # Example of checking conversation history
        elif "last message" in message_lower:
            if len(session['conversation']) > 0:
                last_message = session['conversation'][-1]['user']
                response = f"Your last message was: '{last_message}'"
            else:
                response = "You haven't sent any previous messages."
        
        # Default response for unknown inputs
        else:
            response = query_llm(message)
        
        # Update session with this interaction
        self._update_session(user_id, message, response)        
        return response

# Initialize message handler
handler = MessageHandler()


@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
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
        user_id = message.from_user.id
        incoming_msg = message.text        
        logger.info(f"Received message from {user_id}: {incoming_msg}")        
        # Process the message
        response_text = handler.process_message(user_id, incoming_msg)        
        # Send response
        response_text = handler.process_message(user_id, incoming_msg)
        send_response(message, user_id, response_text)                
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
