import unittest
from unittest.mock import MagicMock, patch, mock_open
from telebot.types import Message, Update # Import Update for webhook testing if needed later

# Import os for use in mock side_effects
import os
# Import the TelegramBotApp class
from RAG_BOT.bot import TelegramBotApp

class TestTelegramBot(unittest.TestCase):

    @patch('RAG_BOT.bot.telebot.TeleBot')
    @patch('RAG_BOT.bot.Flask')
    @patch('RAG_BOT.bot.PdfProcessor')
    @patch('RAG_BOT.bot.HtmProcessor')    
    @patch('RAG_BOT.bot.os') # Patch the entire os module as imported in RAG_BOT.bot
    #@patch('builtins.open', new_callable=mock_open) # Patch builtins.open
    def setUp(self, mock_rag_bot_os,  MockHtmProcessor, MockPdfProcessor, MockFlask, MockTeleBot):
        """Setup method to configure mocks before each test."""
        # Create mock instances for injected dependencies
        # These are received as arguments from the @patch decorators
        self.mock_config = MagicMock()
        self.mock_vector_store_instance = MagicMock()
        self.mock_agent = MagicMock()
        self.mock_handler = MagicMock()
        self.mock_pdf_processor_instance = MockPdfProcessor.return_value
        self.mock_htm_processor_instance = MockHtmProcessor.return_value        

        # Configure mocks as needed for initialization in TelegramBotApp.__init__
        self.mock_config.TELEGRAM_BOT_TOKEN = "dummy_token"
        self.mock_config.DATA_PATH = "dummy_data_path"
        self.mock_config.VECTOR_STORE_PATH = "dummy_vector_store_path"
        self.mock_config.LLM_MODEL_NAME = "dummy_model"
        self.mock_config.RETRIEVER_K = 3
        self.mock_config.WEBHOOK_URL = "dummy_webhook_url"
        self.mock_config.PORT = 5000
        self.mock_config.SEMANTIC_CHUNKING = True # Or False, depending on desired test scenario
        self.mock_config.USER_SESSIONS = {} # For language preference
        # Mock get_user_message to return the default value for simplicity in most tests
        self.mock_config.get_user_message = MagicMock(side_effect=lambda key, default_message: default_message)

        # Configure os mocks for bot.py's internal use (e.g., project_root, uploads dir)
        # These are now attributes of the mocked os module (mock_rag_bot_os)
        self.mock_rag_bot_os = mock_rag_bot_os
        self.mock_rag_bot_os.path = MagicMock() # Mock the path attribute of the mocked os
        # Assign MagicMock instances to the specific os functions we need to control
        self.mock_rag_bot_os.path.abspath = MagicMock()
        self.mock_rag_bot_os.path.join = MagicMock()
        self.mock_rag_bot_os.path.dirname = MagicMock()        
        #self.mock_rag_bot_os.remove = MagicMock()
        self.mock_rag_bot_os.path.exists = MagicMock()        
        self.mock_rag_bot_os.path.exists.return_value = True 
        
        # Define a consistent mocked project root for tests
        self.MOCKED_PROJECT_ROOT = '/mocked_project_root_dir'
        self.MOCKED_UPLOAD_DIR = f'{self.MOCKED_PROJECT_ROOT}/uploads'
        # This will be used to verify calls to open, constructed in _prepare_document_message
        # self.MOCKED_FILE_PATH_FOR_OPEN = f'{self.MOCKED_UPLOAD_DIR}/test_doc.pdf' # Example

        # Configure os.path mocks to ensure self.bot_app.project_root_dir and subsequent paths are mocked
        # This simulates __file__ being in /mocked_project_root_dir/RAG_BOT/bot.py
        self.mock_rag_bot_os.path.dirname.return_value = f"{self.MOCKED_PROJECT_ROOT}/RAG_BOT"
        # Make abspath a pass-through for already "absolute" mock paths
        self.mock_rag_bot_os.path.abspath.side_effect = lambda p: p if p.startswith('/') else f"/{p}"        
        
        def custom_join_side_effect(*args):
            # For os.path.join(self._rag_bot_package_dir, '..') during __init__
            if args == (f"{self.MOCKED_PROJECT_ROOT}/RAG_BOT", '..'):
                return self.MOCKED_PROJECT_ROOT
            # For os.path.join(self.project_root_dir, "uploads") in handle_document
            elif args == (self.MOCKED_PROJECT_ROOT, "uploads"):
                return self.MOCKED_UPLOAD_DIR
            # For os.path.join(upload_dir, file_name) in handle_document
            elif len(args) == 2 and args[0] == self.MOCKED_UPLOAD_DIR:
                return f'{args[0]}/{args[1]}' # e.g., /mocked_project_root_dir/uploads/file.pdf
            # Fallback for any other join, should ideally not be hit by critical paths
            return "/".join(str(arg).strip("/") for arg in args if arg and arg != '.').replace("//","/")

        self.mock_rag_bot_os.path.join.side_effect = custom_join_side_effect

        # Mock the bot instance that would be created by telebot.TeleBot(token)        
        # MockTeleBot.return_value: When your TelegramBotApp is initialized, it executes this line: self.bot = telebot.TeleBot(self.config.TELEGRAM_BOT_TOKEN). 
        # Because telebot.TeleBot is patched, this call actually becomes self.bot = MockTeleBot(self.config.TELEGRAM_BOT_TOKEN). 
        # When a MagicMock (like MockTeleBot) is called as if it were a class constructor or a function, it returns another 
        # MagicMock by default. This returned mock is accessible via its return_value attribute.
        self.mock_bot_instance = MockTeleBot.return_value
        # Mock methods that the bot handlers will call
        self.mock_bot_instance.reply_to = MagicMock()
        self.mock_bot_instance.send_message = MagicMock()
        self.mock_bot_instance.process_new_updates = MagicMock() # For webhook testing if needed
        self.mock_bot_instance.register_message_handler = MagicMock() # Mock the registration method
        self.mock_bot_instance.get_file = MagicMock()
        self.mock_bot_instance.download_file = MagicMock(return_value=b"file_content")

        # Mock the MessageHandler instance and its process_message method
        self.mock_handler.process_message = MagicMock()

        # Mock the Flask app instance and its route method
        self.mock_flask_app_instance = MockFlask.return_value
        # 1. In Flask, you define URL routes using the @app.route('/some/path', methods=['POST']) decorator. This decorator is 
        # essentially a call to the app.route method.
        # 2. The app.route method, when used as a decorator, is called with the route path and potentially other arguments 
        # (/some/path, methods=['POST']). It then returns a second function (the actual decorator). This second function is 
        # then immediately called with the function it's decorating (e.g., your webhook function).
        # 3. This line mocks the route method of your mock Flask instance (self.mock_flask_app_instance).
        # 4. The value assigned to self.mock_flask_app_instance.route is a lambda function designed to mimic the decorator 
        # behavior without actually setting up a route:
        #    - lambda *args, **kwargs:: This outer lambda simulates the first call to @app.route(...). It accepts any positional 
        #       (*args) or keyword (**kwargs) arguments that would normally be passed to route (like the path string and methods 
        #       list). It doesn't use these arguments, but it accepts them so the call doesn't fail..
        #    - : lambda func: func: This is what the outer lambda returns. This inner lambda simulates the actual decorator 
        #       function. It takes one argument, func, which is the function being decorated (in bot.py, this is the webhook 
        #       function). It simply returns func unchanged.        
        # 5. The _setup_webhook_route method in /home/bk_anupam/code/LLM_agents/RAG_BOT/bot.py uses the @self.app.route(...) 
        # decorator. When running unit tests, you don't want the real Flask application to start or try to register routes. 
        # By replacing self.app.route with this mock lambda, you allow the _setup_webhook_route method to execute without error,
        #  but the webhook function is never actually registered with a real Flask routing system. 
        self.mock_flask_app_instance.route = lambda *args, **kwargs: lambda func: func # Mock the decorator

        # Mock vector_store_instance methods
        self.mock_vector_store_instance.get_vectordb.return_value = MagicMock() # Mock the actual vectordb object
        self.mock_vector_store_instance.index_document.return_value = True # Default to successful indexing

        # Mock processor methods
        # Configure mock documents to have a string page_content attribute
        mock_doc_pdf = MagicMock()
        mock_doc_pdf.page_content = "mock pdf content"
        mock_doc_pdf.metadata = {'source': 'test.pdf'} # Add source for consistency
        self.mock_pdf_processor_instance.load_pdf.return_value = [mock_doc_pdf]

        mock_doc_htm = MagicMock()
        mock_doc_htm.page_content = "mock htm content"
        mock_doc_htm.metadata = {'source': 'test.htm'} # Add source for consistency
        self.mock_htm_processor_instance.load_htm.return_value = mock_doc_htm        

        # Instantiate the TelegramBotApp with mocked dependencies
        self.bot_app = TelegramBotApp(
            config=self.mock_config,
            vector_store_instance=self.mock_vector_store_instance,
            agent=self.mock_agent,
            handler=self.mock_handler,
            pdf_processor=self.mock_pdf_processor_instance,
            htm_processor=self.mock_htm_processor_instance
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
        
        # Dummy document for message.document
        self.dummy_document_object = MagicMock()
        self.dummy_document_object.file_id = "test_file_id"
        self.dummy_document_object.mime_type = "application/pdf" # Default for some tests
        self.dummy_document_object.file_name = "test_doc.pdf"

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
            "Welcome to the spiritual chatbot! Ask me questions about the indexed documents, or use /help for commands."
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
            /language <lang> - Set bot language (english or hindi). Example: /language hindi
            /query <your question> [date:YYYY-MM-DD] - Ask a question about the documents. Optionally filter by date.
            You can also just type your question directly.
            """
        )    

    def test_general_message(self):
        """Test handling of a general text message."""
        self.dummy_message.text = "Tell me about the weather."
        # Mock the MessageHandler's process_message to return a predictable response
        # Set up user session for language (default 'en' if not set)
        self.mock_config.USER_SESSIONS = {123: {'language': 'en'}}
        self.mock_handler.process_message.return_value = "The weather is sunny."
        # Call the actual handler method on the bot_app instance
        self.bot_app.handle_all_messages(self.dummy_message)
        # Assert that process_message was called with the correct message
        self.mock_handler.process_message.assert_called_once_with(self.dummy_message, 'en')
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
                vector_store_instance=self.mock_vector_store_instance,
                agent=self.mock_agent,
                handler=self.mock_handler,
                pdf_processor=self.mock_pdf_processor_instance,
                htm_processor=self.mock_htm_processor_instance
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
                vector_store_instance=self.mock_vector_store_instance,
                agent=self.mock_agent,
                handler=self.mock_handler,
                pdf_processor=self.mock_pdf_processor_instance,
                htm_processor=self.mock_htm_processor_instance
            )
        # Optionally, check the exit code
        self.assertEqual(cm.exception.code, 1)
        # Assert that the critical error was logged
        mock_logger.critical.assert_called_once()
        # Check if the log message contains the expected exception string (optional, can be brittle)
        # call_args, _ = mock_logger.critical.call_args
        # self.assertIn("Failed during application startup: TeleBot init failed", call_args[0])

    # --- Tests for /language command ---
    def test_handle_language_command_english(self):
        self.dummy_message.text = "/language english"
        self.mock_config.get_user_message.side_effect = lambda key, default: "Language set to English." if key == "language_set_en" else default
        
        self.bot_app.handle_language_command(self.dummy_message)
        
        self.assertEqual(self.mock_config.USER_SESSIONS[123]['language'], 'en')
        self.mock_bot_instance.reply_to.assert_called_once_with(self.dummy_message, "Language set to English.")

    def test_handle_language_command_hindi(self):
        self.dummy_message.text = "/language hindi"
        self.mock_config.get_user_message.side_effect = lambda key, default: "भाषा हिंदी में सेट कर दी गई है।" if key == "language_set_hi" else default

        self.bot_app.handle_language_command(self.dummy_message)

        self.assertEqual(self.mock_config.USER_SESSIONS[123]['language'], 'hi')
        self.mock_bot_instance.reply_to.assert_called_once_with(self.dummy_message, "भाषा हिंदी में सेट कर दी गई है।")

    def test_handle_language_command_unsupported(self):
        self.dummy_message.text = "/language french"
        self.mock_config.get_user_message.side_effect = lambda key, default: "Unsupported language. Please use 'english' or 'hindi'." if key == "language_unsupported" else default
        
        self.bot_app.handle_language_command(self.dummy_message)
        
        self.assertNotIn(123, self.mock_config.USER_SESSIONS) # Or check it's unchanged
        self.mock_bot_instance.reply_to.assert_called_once_with(self.dummy_message, "Unsupported language. Please use 'english' or 'hindi'.")

    def test_handle_language_command_no_argument(self):
        self.dummy_message.text = "/language"
        self.mock_config.get_user_message.side_effect = lambda key, default: "Usage: /language <language>\nSupported languages: english, hindi" if key == "language_usage_help" else default

        self.bot_app.handle_language_command(self.dummy_message)
        
        self.mock_bot_instance.reply_to.assert_called_once_with(self.dummy_message, "Usage: /language <language>\nSupported languages: english, hindi")

    # --- Tests for handle_document ---
    def _prepare_document_message(self, mime_type="application/pdf", file_name="test.pdf"):
        self.dummy_message.document = MagicMock()
        self.dummy_message.document.file_id = "file_id_123"
        self.dummy_message.document.mime_type = mime_type
        # This file_name is crucial. It will be used by custom_join_side_effect
        self.dummy_message.document.file_name = file_name
        
        mock_file_info = MagicMock()
        mock_file_info.file_path = "telegram/files/doc.pdf" # This is just a dummy for bot.get_file
        self.mock_bot_instance.get_file.return_value = mock_file_info
        self.mock_bot_instance.download_file.return_value = b"some file data"        

        # This will be the path that 'open' is expected to be called with
        self.MOCKED_FILE_PATH_FOR_OPEN = f"{self.MOCKED_UPLOAD_DIR}/{file_name}"

        mock_doc_pdf = MagicMock()
        mock_doc_pdf.page_content = "pdf content for " + file_name
        mock_doc_pdf.metadata = {'source': file_name}
        # Ensure load_pdf is reset and configured for this specific test preparation
        self.mock_pdf_processor_instance.load_pdf.reset_mock()
        self.mock_pdf_processor_instance.load_pdf.return_value = [mock_doc_pdf]

        mock_doc_htm = MagicMock()
        mock_doc_htm.page_content = "htm content for " + file_name
        mock_doc_htm.metadata = {'source': file_name}
        self.mock_htm_processor_instance.load_htm.reset_mock()
        self.mock_htm_processor_instance.load_htm.return_value = mock_doc_htm
        
        self.mock_vector_store_instance.index_document.reset_mock(return_value=True)        


    @patch('RAG_BOT.bot.os.path.exists') 
    @patch('RAG_BOT.bot.detect_document_language') 
    @patch('RAG_BOT.bot.os.remove') # Patch os.remove
    @patch('builtins.open', new_callable=mock_open) # Patch builtins.open
    @patch('RAG_BOT.bot.os.makedirs') 
    def test_handle_document_pdf_success(self, mock_os_makedirs, mock_bot_open, mock_os_remove,
                                         mock_detect_language, mock_os_path_exists):        
        self._prepare_document_message(mime_type="application/pdf", file_name="test_doc.pdf")        
        mock_os_makedirs.return_value = None 
        mock_os_remove.return_value = None 
        mock_detect_language.return_value = 'en'         
        mock_os_path_exists.return_value = True                        

        self.bot_app.handle_document(self.dummy_message)
        
        mock_bot_open.assert_called_once_with(self.MOCKED_FILE_PATH_FOR_OPEN, 'wb')
        self.mock_pdf_processor_instance.load_pdf.assert_called_once_with(self.MOCKED_FILE_PATH_FOR_OPEN)
        mock_detect_language.assert_called_once() # Should be called with documents from load_pdf
        self.mock_vector_store_instance.index_document.assert_called_once()
        self.mock_bot_instance.reply_to.assert_called_with(self.dummy_message, "Document 'test_doc.pdf' uploaded and indexed successfully.")
        mock_os_remove.assert_called_once_with(self.MOCKED_FILE_PATH_FOR_OPEN)


    @patch('RAG_BOT.bot.os.path.exists') 
    @patch('RAG_BOT.bot.detect_document_language') 
    @patch('RAG_BOT.bot.os.remove') 
    @patch('builtins.open', new_callable=mock_open) 
    @patch('RAG_BOT.bot.os.makedirs') 
    def test_handle_document_htm_success(self, mock_os_makedirs, mock_bot_open, mock_os_remove,
                                         mock_detect_language, mock_os_path_exists):
        mock_os_makedirs.return_value = None 
        mock_os_remove.return_value = None 
        mock_detect_language.return_value = 'en'         
        mock_os_path_exists.return_value = True                        
        self._prepare_document_message(mime_type="text/html", file_name="test_page.htm")        

        self.bot_app.handle_document(self.dummy_message)

        mock_bot_open.assert_called_once_with(self.MOCKED_FILE_PATH_FOR_OPEN, 'wb')
        self.mock_htm_processor_instance.load_htm.assert_called_once_with(self.MOCKED_FILE_PATH_FOR_OPEN)
        self.mock_vector_store_instance.index_document.assert_called_once()
        self.mock_bot_instance.reply_to.assert_called_with(self.dummy_message, "Document 'test_page.htm' uploaded and indexed successfully.")
        mock_os_remove.assert_called_once_with(self.MOCKED_FILE_PATH_FOR_OPEN)

    @patch('RAG_BOT.bot.os.path.exists') 
    @patch('RAG_BOT.bot.detect_document_language') 
    @patch('RAG_BOT.bot.os.remove') 
    @patch('builtins.open', new_callable=mock_open) 
    @patch('RAG_BOT.bot.os.makedirs') 
    def test_handle_document_octet_stream_pdf_success(self, mock_os_makedirs, mock_bot_open, mock_os_remove,
                                         mock_detect_language, mock_os_path_exists):
        mock_os_makedirs.return_value = None 
        mock_os_remove.return_value = None 
        mock_detect_language.return_value = 'en'         
        mock_os_path_exists.return_value = True  
        self._prepare_document_message(mime_type="application/octet-stream", file_name="document.pdf")        

        self.bot_app.handle_document(self.dummy_message)
        
        mock_bot_open.assert_called_once_with(self.MOCKED_FILE_PATH_FOR_OPEN, 'wb')
        self.mock_pdf_processor_instance.load_pdf.assert_called_once_with(self.MOCKED_FILE_PATH_FOR_OPEN)
        self.mock_bot_instance.reply_to.assert_called_with(self.dummy_message, "Document 'document.pdf' uploaded and indexed successfully.")
        mock_os_remove.assert_called_once_with(self.MOCKED_FILE_PATH_FOR_OPEN)

    @patch('RAG_BOT.bot.os.path.exists') 
    @patch('RAG_BOT.bot.detect_document_language') 
    @patch('RAG_BOT.bot.os.remove') 
    @patch('builtins.open', new_callable=mock_open) 
    @patch('RAG_BOT.bot.os.makedirs') 
    def test_handle_document_octet_stream_htm_success(self, mock_os_makedirs, mock_bot_open, mock_os_remove,
                                         mock_detect_language, mock_os_path_exists):
        mock_os_makedirs.return_value = None 
        mock_os_remove.return_value = None 
        mock_detect_language.return_value = 'en'         
        mock_os_path_exists.return_value = True
        self._prepare_document_message(mime_type="application/octet-stream", file_name="archive.html")

        self.bot_app.handle_document(self.dummy_message)

        mock_bot_open.assert_called_once_with(self.MOCKED_FILE_PATH_FOR_OPEN, 'wb')
        self.mock_htm_processor_instance.load_htm.assert_called_once_with(self.MOCKED_FILE_PATH_FOR_OPEN)
        self.mock_bot_instance.reply_to.assert_called_with(self.dummy_message, "Document 'archive.html' uploaded and indexed successfully.")
        mock_os_remove.assert_called_once_with(self.MOCKED_FILE_PATH_FOR_OPEN)


    def test_handle_document_unsupported_mime_type(self):
        self._prepare_document_message(mime_type="application/zip", file_name="archive.zip")
        
        self.bot_app.handle_document(self.dummy_message)
        
        self.mock_bot_instance.reply_to.assert_called_with(self.dummy_message, "Unsupported file type (application/zip).")
        self.mock_rag_bot_os.remove.assert_not_called()

    def test_handle_document_octet_stream_unsupported_extension(self):
        self._prepare_document_message(mime_type="application/octet-stream", file_name="data.dat")

        self.bot_app.handle_document(self.dummy_message)

        self.mock_bot_instance.reply_to.assert_called_with(self.dummy_message, "Unsupported file type or unable to determine type from 'data.dat'.")
        self.mock_rag_bot_os.remove.assert_not_called()

    @patch('RAG_BOT.bot.os.path.exists') 
    @patch('RAG_BOT.bot.detect_document_language') 
    @patch('RAG_BOT.bot.os.remove') 
    @patch('builtins.open', new_callable=mock_open) 
    @patch('RAG_BOT.bot.os.makedirs')
    def test_handle_document_load_fails(self, mock_os_makedirs, mock_bot_open, mock_os_remove,
                                         mock_detect_language, mock_os_path_exists):
        mock_os_makedirs.return_value = None 
        mock_os_remove.return_value = None 
        mock_detect_language.return_value = 'en'         
        mock_os_path_exists.return_value = True
        self._prepare_document_message(mime_type="application/pdf", file_name="empty.pdf")
        self.mock_pdf_processor_instance.load_pdf.return_value = [] # Simulate load failure        

        self.bot_app.handle_document(self.dummy_message)

        self.mock_bot_instance.reply_to.assert_called_with(self.dummy_message, "Could not load content from 'empty.pdf'.")
        mock_os_remove.assert_not_called() # File should not be removed if load fails
        mock_detect_language.assert_not_called() # Language detection should not be called if load fails
        self.mock_vector_store_instance.index_document.assert_not_called() # Indexing should not be attempted if load fails

    @patch('RAG_BOT.bot.os.path.exists') 
    @patch('RAG_BOT.bot.detect_document_language') 
    @patch('RAG_BOT.bot.os.remove') 
    @patch('builtins.open', new_callable=mock_open) 
    @patch('RAG_BOT.bot.os.makedirs')
    def test_handle_document_indexing_fails(self, mock_os_makedirs, mock_bot_open, mock_os_remove,
                                         mock_detect_language, mock_os_path_exists):
        mock_os_makedirs.return_value = None 
        mock_os_remove.return_value = None 
        mock_detect_language.return_value = 'en'         
        mock_os_path_exists.return_value = True
        self._prepare_document_message(mime_type="application/pdf", file_name="noindex.pdf")
        self.mock_vector_store_instance.index_document.return_value = False # Simulate indexing failure        

        self.bot_app.handle_document(self.dummy_message)

        self.mock_bot_instance.reply_to.assert_called_with(self.dummy_message, "Document 'noindex.pdf' was not indexed (possibly already exists or an error occurred).")
        mock_os_remove.assert_not_called() # File should not be removed if indexing fails

    def test_handle_document_no_document_in_message(self):
        self.dummy_message.document = None # Ensure no document is attached

        self.bot_app.handle_document(self.dummy_message)

        self.mock_bot_instance.reply_to.assert_called_with(self.dummy_message, "No document provided.")
        self.mock_rag_bot_os.remove.assert_not_called()

    @patch('RAG_BOT.bot.logger')
    def test_handle_document_processing_exception(self, mock_logger):
        self._prepare_document_message(mime_type="application/pdf", file_name="error.pdf")
        self.mock_pdf_processor_instance.load_pdf.side_effect = Exception("PDF processing error")
        self.mock_rag_bot_os.path.exists.return_value = True # File was saved before error

        self.bot_app.handle_document(self.dummy_message)

        self.mock_bot_instance.reply_to.assert_called_with(self.dummy_message, "Sorry, I encountered an error processing your document.")
        mock_logger.error.assert_called()
        self.mock_rag_bot_os.remove.assert_not_called() # File should not be removed if an error occurs during processing

    @patch('RAG_BOT.bot.os.path.exists') 
    @patch('RAG_BOT.bot.detect_document_language') 
    @patch('RAG_BOT.bot.os.remove') 
    @patch('builtins.open', new_callable=mock_open) 
    @patch('RAG_BOT.bot.os.makedirs')
    def test_handle_document_cleanup_file_not_exists_after_processing(self, mock_os_makedirs, mock_bot_open, mock_os_remove,
                                         mock_detect_language, mock_os_path_exists):    
        mock_os_makedirs.return_value = None 
        mock_os_remove.return_value = None 
        mock_detect_language.return_value = 'en'         
        # Simulate successful processing, but file doesn't exist for removal (e.g., already cleaned up by another process)        
        mock_os_path_exists.return_value = False
        self._prepare_document_message(mime_type="application/pdf", file_name="gone.pdf")        

        self.bot_app.handle_document(self.dummy_message)

        self.mock_bot_instance.reply_to.assert_called_with(self.dummy_message, "Document 'gone.pdf' uploaded and indexed successfully.")
        mock_os_remove.remove.assert_not_called() # Removal shouldn't be attempted if file doesn't exist


if __name__ == '__main__':
    unittest.main()
