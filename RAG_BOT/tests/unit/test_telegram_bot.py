import unittest
from unittest.mock import MagicMock, patch
from telebot.types import Message, Update # Import Update for webhook testing if needed later

# Add the project root to the Python path
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Import the TelegramBotApp class
from RAG_BOT.bot import TelegramBotApp

class TestTelegramBot(unittest.TestCase):

    @patch('RAG_BOT.bot.telebot.TeleBot') # Patch the TeleBot class
    @patch('RAG_BOT.bot.Flask') # Patch Flask app initialization
    def setUp(self, MockFlask, MockTeleBot):
        """Setup method to configure mocks before each test."""
        # Create mock instances for injected dependencies
        self.mock_config = MagicMock()
        self.mock_vectordb = MagicMock()
        self.mock_agent = MagicMock()
        self.mock_handler = MagicMock()

        # Configure mocks as needed for initialization in TelegramBotApp.__init__
        self.mock_config.TELEGRAM_BOT_TOKEN = "dummy_token"
        self.mock_config.DATA_PATH = "dummy_data_path"
        self.mock_config.VECTOR_STORE_PATH = "dummy_vector_store_path"
        self.mock_config.LLM_MODEL_NAME = "dummy_model"
        self.mock_config.RETRIEVER_K = 3
        self.mock_config.WEBHOOK_URL = "dummy_webhook_url"
        self.mock_config.PORT = 5000
        self.mock_config.SEMANTIC_CHUNKING = True # Or False, depending on desired test scenario

        # Mock the bot instance that would be created by telebot.TeleBot(token)
        self.mock_bot_instance = MockTeleBot.return_value
        # Mock methods that the bot handlers will call
        self.mock_bot_instance.reply_to = MagicMock()
        self.mock_bot_instance.send_message = MagicMock()
        self.mock_bot_instance.process_new_updates = MagicMock() # For webhook testing if needed
        self.mock_bot_instance.register_message_handler = MagicMock() # Mock the registration method

        # Mock the MessageHandler instance and its process_message method
        self.mock_handler.process_message = MagicMock()

        # Mock the Flask app instance and its route method
        self.mock_flask_app_instance = MockFlask.return_value
        self.mock_flask_app_instance.route = lambda *args, **kwargs: lambda func: func # Mock the decorator

        # Instantiate the TelegramBotApp with mocked dependencies
        self.bot_app = TelegramBotApp(
            config=self.mock_config,
            vectordb=self.mock_vectordb,
            agent=self.mock_agent,
            handler=self.mock_handler
        )

        # Create a dummy message object to simulate incoming messages
        # This object needs attributes that the handlers access
        self.dummy_message = MagicMock(spec=Message)
        self.dummy_message.text = "" # Default text
        # Explicitly mock the from_user attribute
        self.dummy_message.from_user = MagicMock()
        self.dummy_message.chat = MagicMock()        
        self.dummy_message.from_user.id = 123 # Dummy user ID
        self.dummy_message.chat.id = 456 # Dummy chat ID
        self.dummy_message.document = None # For document handling tests

    def tearDown(self):
        """Cleanup method after each test."""
        # Reset mocks if necessary, though patch usually handles this per test
        pass

    def test_start_command(self):
        """Test handling of the /start command."""
        self.dummy_message.text = "/start"
        # Call the actual handler method on the bot_app instance
        self.bot_app.send_welcome(self.dummy_message)
        # Assert that reply_to was called on the mocked bot instance with the correct arguments
        self.mock_bot_instance.reply_to.assert_called_once_with(
            self.dummy_message,
            "Welcome to the RAG Bot! Ask me questions about the indexed documents, or use /help for commands."
        )

    def test_help_command(self):
        """Test handling of the /help command."""
        self.dummy_message.text = "/help"
        # Call the actual handler method on the bot_app instance
        self.bot_app.send_help(self.dummy_message)
        # Assert that reply_to was called on the mocked bot instance with the correct arguments
        self.mock_bot_instance.reply_to.assert_called_once_with(
            self.dummy_message,
            """
            Available Commands:
            /start - Show welcome message.
            /help - Show this help message.
            /query <your question> [date:YYYY-MM-DD] - Ask a question about the documents. Optionally filter by date.
            You can also just type your question directly.
            """
        )    

    def test_general_message(self):
        """Test handling of a general text message."""
        self.dummy_message.text = "Tell me about the weather."
        # Mock the MessageHandler's process_message to return a predictable response
        self.mock_handler.process_message.return_value = "The weather is sunny."
        # Call the actual handler method on the bot_app instance
        self.bot_app.handle_all_messages(self.dummy_message)
        # Assert that process_message was called with the correct message
        self.mock_handler.process_message.assert_called_once_with(self.dummy_message)
        # Assert that reply_to was called on the mocked bot instance with the correct arguments
        self.mock_bot_instance.reply_to.assert_called_once_with(
            self.dummy_message,
            "The weather is sunny."
        )

    def test_send_response_long_message(self):
        """Test send_response with a message longer than Telegram's limit."""
        long_response = "A" * 5000 # Longer than 4096
        user_id = self.dummy_message.from_user.id
        chat_id = self.dummy_message.chat.id

        # Call the actual send_response method on the bot_app instance
        self.bot_app.send_response(self.dummy_message, user_id, long_response)

        # Assert that reply_to was called with the first chunk
        self.mock_bot_instance.reply_to.assert_called_once_with(
            self.dummy_message,
            long_response[:4096]
        )
        # Assert that send_message was called with the subsequent chunks
        # There should be one more chunk
        self.mock_bot_instance.send_message.assert_called_once_with(
            chat_id,
            long_response[4096:]
        )

    def test_send_response_empty_message(self):
        """Test send_response with an empty message."""
        empty_response = ""
        user_id = self.dummy_message.from_user.id

        # Call the actual send_response method on the bot_app instance
        self.bot_app.send_response(self.dummy_message, user_id, empty_response)

        # Assert that reply_to was called with the fallback message
        self.mock_bot_instance.reply_to.assert_called_once_with(
            self.dummy_message,
            "Sorry, I could not generate a response."
        )
        # Ensure send_message was not called for an empty response
        self.mock_bot_instance.send_message.assert_not_called()

    @patch('RAG_BOT.bot.logger') # Patch the logger within the bot module
    def test_init_missing_token(self, mock_logger):
        """Test initialization failure when TELEGRAM_BOT_TOKEN is missing."""
        # Reset the relevant mock config value for this specific test
        self.mock_config.TELEGRAM_BOT_TOKEN = None
        # Assert that initializing the app raises SystemExit (due to exit(1))
        with self.assertRaises(SystemExit) as cm:
            TelegramBotApp(
                config=self.mock_config,
                vectordb=self.mock_vectordb,
                agent=self.mock_agent,
                handler=self.mock_handler
            )
        # Optionally, check the exit code
        self.assertEqual(cm.exception.code, 1)
        # Assert that the specific error was logged
        mock_logger.error.assert_called_once_with(
            "TELEGRAM_BOT_TOKEN is not set. Please set it in your environment variables."
        )

    @patch('RAG_BOT.bot.telebot.TeleBot') # Re-patch TeleBot specifically for this test
    @patch('RAG_BOT.bot.logger') # Patch the logger
    def test_init_general_exception(self, mock_logger, MockTeleBot):
        """Test initialization failure due to a general exception (e.g., TeleBot error)."""
        # Configure the TeleBot mock to raise an exception when called
        MockTeleBot.side_effect = Exception("TeleBot init failed")

        # Assert that initializing the app raises SystemExit
        with self.assertRaises(SystemExit) as cm:
            TelegramBotApp(
                config=self.mock_config,
                vectordb=self.mock_vectordb,
                agent=self.mock_agent,
                handler=self.mock_handler
            )
        # Optionally, check the exit code
        self.assertEqual(cm.exception.code, 1)
        # Assert that the critical error was logged
        mock_logger.critical.assert_called_once()
        # Check if the log message contains the expected exception string (optional, can be brittle)
        # call_args, _ = mock_logger.critical.call_args
        # self.assertIn("Failed during application startup: TeleBot init failed", call_args[0])


if __name__ == '__main__':
    unittest.main()
