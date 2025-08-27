import os
import sys
import shutil
import unittest
import threading
import time
import requests
import json
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any, Optional
from telebot import types # Import types for message matching
from flask import request

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Ensure RAG_BOT modules are importable
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.logger import logger
from RAG_BOT.src.persistence.vector_store import VectorStore
from RAG_BOT.src.agent.graph_builder import build_agent
from RAG_BOT.src.services.message_handler import MessageHandler
from RAG_BOT.src.telegram.bot import TelegramBotApp # Import the app class
from RAG_BOT.src import utils
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# --- Global variables for mock communication ---
# Using a simple list to capture responses. Ensure thread safety if needed.
mock_bot_responses: List[str] = []
mock_bot_lock = threading.Lock()

# --- Mock TeleBot ---
class MockTeleBot:
    """Mocks telebot.TeleBot to capture outgoing messages and simulate handler dispatch."""
    def __init__(self, token, *args, **kwargs):
        logger.info(f"MockTeleBot initialized with token: {token[:5]}...")
        self.token = token
        # Store handlers registered via register_message_handler
        self.handlers = []

    def reply_to(self, message, text, *args, **kwargs):
        logger.info(f"MockTeleBot captured reply_to: {text[:100]}...")
        with mock_bot_lock:
            mock_bot_responses.append(text)

    def send_message(self, chat_id, text, *args, **kwargs):
        logger.info(f"MockTeleBot captured send_message to {chat_id}: {text[:100]}...")
        with mock_bot_lock:
            mock_bot_responses.append(text)

    def set_webhook(self, *args, **kwargs):
        logger.info("MockTeleBot: set_webhook called")
        return True # Simulate success

    def remove_webhook(self, *args, **kwargs):
        logger.info("MockTeleBot: remove_webhook called")
        return True # Simulate success

    def get_file(self, file_id):
        logger.warning(f"MockTeleBot: get_file called for {file_id} - returning dummy")
        mock_file = MagicMock()
        mock_file.file_path = "dummy/path/file.pdf"
        return mock_file

    def download_file(self, file_path):
        logger.warning(f"MockTeleBot: download_file called for {file_path} - returning dummy bytes")
        return b"dummy pdf content"

    def register_message_handler(self, callback, commands=None, regexp=None, content_types=None, func=None, **kwargs):
        """Stores the handler function and its filters."""
        logger.info(f"MockTeleBot: Registering handler: {callback.__name__} (Commands: {commands}, ContentTypes: {content_types}, Func: {func is not None})")
        self.handlers.append({
            'callback': callback,
            'filters': {
                'commands': commands,
                'regexp': regexp,
                'content_types': content_types or ['text'], # Default to text if None
                'func': func
            }
        })

    def process_new_updates(self, updates: List[types.Update]):
        """
        Simulates pyTelegramBotAPI's update processing by finding and calling
        the appropriate registered handler.
        """
        logger.info(f"MockTeleBot: process_new_updates called with {len(updates)} update(s)")
        for update in updates:
            if update.message:
                message = update.message
                logger.info(f"MockTeleBot: Processing message update (ID: {message.message_id}, Text: '{message.text[:50]}...', Type: {message.content_type})")
                handler_called = False
                # Iterate through registered handlers to find a match
                for handler_reg in self.handlers:
                    filters = handler_reg['filters']
                    callback = handler_reg['callback']

                    # --- Basic Filter Matching Logic ---
                    match = True

                    # Match content_type
                    if filters['content_types'] and message.content_type not in filters['content_types']:
                        match = False

                    # Match commands (only if content_type is text)
                    if match and filters['commands'] and message.content_type == 'text':
                        is_command = False
                        if message.text and message.text.startswith('/'):
                            command = message.text.split()[0][1:]
                            if command in filters['commands']:
                                is_command = True
                        if not is_command:
                             match = False # Command filter exists, but message isn't that command

                    # Match func (lambda filter)
                    if match and filters['func']:
                        if not filters['func'](message):
                            match = False

                    # TODO: Add regexp matching if needed

                    # If all filters pass, call the handler
                    if match:
                        logger.info(f"MockTeleBot: Matched handler '{callback.__name__}'. Calling it.")
                        try:
                            callback(message)
                            handler_called = True
                            break # Stop after first match (like pyTelegramBotAPI often does)
                        except Exception as e:
                            logger.error(f"MockTeleBot: Error executing handler '{callback.__name__}': {e}", exc_info=True)
                            # Decide if you want to stop or continue checking other handlers
                            break
                if not handler_called:
                     logger.warning(f"MockTeleBot: No suitable handler found for message (ID: {message.message_id}, Text: '{message.text[:50]}...')")

            # TODO: Add handling for other update types (callback_query, etc.) if needed
            else:
                 logger.debug(f"MockTeleBot: Skipping update without message: {update}")


# --- Test Class ---
class TestEndToEnd(unittest.TestCase):

    bot_thread: Optional[threading.Thread] = None
    bot_app_instance: Optional[TelegramBotApp] = None
    webhook_url: Optional[str] = None
    config: Optional[Config] = None
    test_vector_store_dir: Optional[str] = None
    patcher: Optional[Any] = None # To hold the patch object

    @classmethod
    def setUpClass(cls):
        """Setup method called once before all tests."""
        cls.config = Config()
        cls.setup_test_vector_store() # Create and index test data

        # --- Start Mocking ---
        # Patch telebot.TeleBot where it's imported in RAG_BOT.bot
        cls.patcher = patch('RAG_BOT.bot.telebot.TeleBot', MockTeleBot)
        cls.patcher.start()
        logger.info("Telebot patch started.")

        # --- Initialize Bot Components ---
        # These need to be created *after* the patch is active
        try:
            logger.info("Initializing VectorStore for E2E test...")
            vector_store_instance = VectorStore(persist_directory=cls.test_vector_store_dir)
            vectordb = vector_store_instance.get_vectordb()
            if vectordb is None:
                raise ValueError("Failed to initialize vectordb for E2E test.")
            logger.info("VectorStore initialized.")

            logger.info("Initializing RAG agent for E2E test...")
            agent = build_agent(vectordb=vectordb, config_instance=cls.config)
            logger.info("RAG agent initialized.")

            logger.info("Initializing MessageHandler for E2E test...")
            handler = MessageHandler(agent=agent, config=cls.config)
            logger.info("MessageHandler initialized.")

            logger.info("Initializing TelegramBotApp for E2E test...")
            # Pass dependencies explicitly
            cls.bot_app_instance = TelegramBotApp(
                config=cls.config,
                vectordb=vectordb, # Pass the Chroma instance directly
                agent=agent,
                handler=handler
            )
            # Add a shutdown route for testing
            @cls.bot_app_instance.app.route('/shutdown', methods=['POST'])
            def shutdown():
                logger.warning("Shutdown route called, attempting to stop server.")
                func = request.environ.get('werkzeug.server.shutdown')
                if func is None:
                    logger.error('Not running with the Werkzeug Server or shutdown unavailable.')
                    # Fallback or error handling needed here if not using Werkzeug
                    # For simplicity, we might rely on thread termination, but graceful shutdown is better.
                    return 'Could not shut down server gracefully.', 500
                func()
                return 'Server shutting down...'

            logger.info("TelegramBotApp initialized.")

        except Exception as e:
            logger.critical(f"Failed to initialize bot components during setUpClass: {e}", exc_info=True)
            cls.tearDownClass() # Attempt cleanup
            raise # Re-raise the exception to fail the setup

        # --- Start Bot in Thread ---
        cls.webhook_url = f"http://127.0.0.1:{cls.config.PORT}/{cls.config.TELEGRAM_BOT_TOKEN}"
        cls.bot_thread = threading.Thread(
            target=cls.bot_app_instance.run, # Use the instance's run method
            daemon=True # Allows main thread to exit even if this thread is running
        )
        cls.bot_thread.start()
        logger.info(f"Bot thread started. Waiting for server at {cls.webhook_url}...")

        # --- Wait for Server Readiness ---
        max_wait = 30 # seconds
        start_time = time.time()
        server_ready = False
        while time.time() - start_time < max_wait:
            try:
                # Use a simple GET request to check if the base URL is responding
                # Note: The webhook URL expects POST, so GET might 404/405, but indicates the server is up.
                response = requests.get(f"http://127.0.0.1:{cls.config.PORT}/", timeout=1)
                # Check for any response, even errors like 404/405, means the server is listening
                if response.status_code:
                    logger.info(f"Server responded with status {response.status_code}. Assuming ready.")
                    server_ready = True
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)
            except Exception as e:
                 logger.warning(f"Error checking server readiness: {e}")
                 time.sleep(0.5)

        if not server_ready:
            logger.error("Server did not become ready within the timeout period.")
            cls.tearDownClass() # Attempt cleanup
            raise ConnectionError("Flask server did not start in time for E2E tests.")

        logger.info("setUpClass completed.")


    @classmethod
    def tearDownClass(cls):
        """Teardown method called once after all tests."""
        logger.info("Starting tearDownClass...")
        if cls.bot_thread and cls.bot_thread.is_alive():
            logger.info("Attempting to shut down Flask server...")
            try:
                # Send shutdown request
                shutdown_url = f"http://127.0.0.1:{cls.config.PORT}/shutdown"
                requests.post(shutdown_url, timeout=5)
                logger.info("Shutdown request sent.")
            except Exception as e:
                logger.error(f"Failed to send shutdown request: {e}")
                # If shutdown request fails, thread might still be running.

            # Wait for the thread to finish
            cls.bot_thread.join(timeout=10)
            if cls.bot_thread.is_alive():
                logger.warning("Bot thread did not terminate gracefully after shutdown request and join timeout.")
                # Consider more forceful termination if necessary, but can leave resources hanging.
            else:
                logger.info("Bot thread terminated.")
        else:
             logger.info("Bot thread was not running or already stopped.")

        # Stop the patcher
        if cls.patcher:
            cls.patcher.stop()
            logger.info("Telebot patch stopped.")
            cls.patcher = None

        # Clean up test vector store
        cls.delete_test_vector_store()
        logger.info("tearDownClass completed.")

    @classmethod
    def setup_test_vector_store(cls):
        """Creates a clean vector store and indexes test PDFs."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_vector_store_dir = os.path.join(current_dir, "..", "test_vector_store")
        pdf_dir = os.path.join(current_dir, "..", "data")

        cls.delete_test_vector_store() # Ensure clean start
        os.makedirs(cls.test_vector_store_dir, exist_ok=True)
        logger.info(f"Setting up test vector store for E2E in: {cls.test_vector_store_dir}")

        test_vector_store = VectorStore(persist_directory=cls.test_vector_store_dir)
        pdf_files = [
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if f.endswith(".pdf")
        ]
        if not pdf_files:
             logger.warning(f"No PDF files found in {pdf_dir} for E2E indexing.")
             return

        for pdf_file in pdf_files:
            logger.info(f"Indexing E2E test file: {pdf_file}")
            # Use config for semantic chunking setting
            test_vector_store.build_index(pdf_file, semantic_chunk=cls.config.SEMANTIC_CHUNKING)
        logger.info("E2E test vector store setup complete.")


    @classmethod
    def delete_test_vector_store(cls):
        """Deletes the test vector store directory if it exists."""
        if cls.test_vector_store_dir and os.path.exists(cls.test_vector_store_dir):
            try:
                shutil.rmtree(cls.test_vector_store_dir)
                logger.info(f"Deleted E2E test vector store at: {cls.test_vector_store_dir}")
            except Exception as e:
                logger.error(f"Error deleting E2E test vector store: {e}")


    def setUp(self):
        """Called before each test method."""
        # Clear responses before each test
        with mock_bot_lock:
            mock_bot_responses.clear()
        # Ensure server is still alive (optional sanity check)
        self.assertTrue(self.bot_thread and self.bot_thread.is_alive(), "Bot thread is not alive at start of test.")


    def _send_message(self, text: str, user_id: int = 12345, chat_id: int = 12345):
        """Simulates sending a message to the bot's webhook."""
        if not self.webhook_url:
            self.fail("Webhook URL not set.")

        # Construct a basic Telegram Update JSON structure
        update_data = {
            "update_id": int(time.time() * 1000), # Unique enough for testing
            "message": {
                "message_id": int(time.time() * 1000) + 1,
                "from": {"id": user_id, "is_bot": False, "first_name": "Test", "last_name": "User", "username": "testuser"},
                "chat": {"id": chat_id, "type": "private", "first_name": "Test", "last_name": "User", "username": "testuser"},
                "date": int(time.time()),
                "text": text
            }
        }
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(self.webhook_url, headers=headers, json=update_data, timeout=20) # Increased timeout for agent processing
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            logger.info(f"Sent message to webhook: '{text[:50]}...'. Response status: {response.status_code}")
            # Give the bot thread some time to process and call the mock reply
            time.sleep(5) # Adjust as needed based on agent complexity
        except requests.exceptions.RequestException as e:
            self.fail(f"Failed to send message to webhook {self.webhook_url}: {e}")


    def _get_latest_response(self) -> Optional[str]:
        """Retrieves the latest response captured by the mock bot."""
        with mock_bot_lock:
            if mock_bot_responses:
                # Return the last captured response
                return mock_bot_responses[-1]
            return None


    def evaluate_response_with_llm(self, query: str, response: str) -> bool:
        """Uses an LLM to judge the quality of the agent's response."""
        # Note: Context is not easily available in E2E like in integration tests.
        # We judge based on query and response only for simplicity here.
        judge_llm = ChatGoogleGenerativeAI(model=Config.JUDGE_LLM_MODEL_NAME, temperature=0.0)
        # Simplified judge prompt for E2E (without context)
        judge_prompt_template_str = """
        You are an impartial judge evaluating the quality of an AI assistant's response to a user query.
        Consider if the response directly addresses the query and is informative.
        User Query:
        {query}

        Assistant Response:
        {response}

        Is the assistant's answer relevant and informative for the user query?
        Respond with only 'PASS' or 'FAIL'.
        """
        judge_prompt = judge_prompt_template_str.format(query=query, response=response)
        try:
            evaluation = judge_llm.invoke([HumanMessage(content=judge_prompt)]).content.strip().upper()
            logger.info(f"LLM Judge Evaluation for E2E query '{query[:50]}...': {evaluation}")
            return evaluation == 'PASS'
        except Exception as e:
            logger.error(f"LLM Judge call failed during E2E test: {e}")
            return False # Fail the test if judge fails


    # --- Test Cases ---

    def test_successful_query_e2e(self):
        """Tests a query expected to succeed with context retrieval."""
        query = "What is the title of the murli from 1969-01-23?"
        self._send_message(query)
        response = self._get_latest_response()

        self.assertIsNotNone(response, "Bot did not send any response.")
        logger.info(f"Raw response received: {response}")

        # Check for expected content (case-insensitive substring)
        self.assertIn("the ashes are to remind you of the stage", response.lower(),
                      f"Expected content not found in answer: {response}")

        # Evaluate with LLM Judge
        self.assertTrue(self.evaluate_response_with_llm(query, response),
                        f"LLM Judge evaluation failed for successful query. Response: {response}")


    def test_insufficient_context_query_e2e(self):
        """Tests a query expected to fail due to lack of relevant context."""
        query = "Can you summarize the murli from 1950-01-18?" # Date likely not in test data
        self._send_message(query)
        response = self._get_latest_response()

        self.assertIsNotNone(response, "Bot did not send any response for insufficient context query.")
        logger.info(f"Raw response received (insufficient context): {response}")

        # Check for "cannot find" message
        self.assertTrue(
            "cannot be found" in response.lower() or "cannot find" in response.lower(),
            f"Agent did not return a 'cannot find' message: {response}"
        )

        # Evaluate with LLM Judge (optional for 'cannot find', but can check relevance)
        # self.assertTrue(self.evaluate_response_with_llm(query, response),
        #                 f"LLM Judge evaluation failed for insufficient context query. Response: {response}")


    def test_general_knowledge_query_e2e(self):
        """Tests a general knowledge question not requiring retrieval."""
        query = "What is the capital of France?"
        self._send_message(query)
        response = self._get_latest_response()

        self.assertIsNotNone(response, "Bot did not send any response for general query.")
        logger.info(f"Raw response received (general query): {response}")

        # Check that it didn't say "cannot find"
        answer_lower = response.lower()
        self.assertNotIn("cannot be found", answer_lower)
        self.assertNotIn("cannot find", answer_lower)
        # Check for expected content (case-insensitive substring)
        self.assertIn("paris", answer_lower, f"Expected 'Paris' not found in answer: {response}")

        # Evaluate with LLM Judge
        self.assertTrue(self.evaluate_response_with_llm(query, response),
                        f"LLM Judge evaluation failed for general query. Response: {response}")


if __name__ == "__main__":
    # Ensure Flask and other dependencies are importable when running directly
    # This might require adjusting PYTHONPATH or running with `python -m unittest ...`
    # Add request explicitly for the shutdown route
    from flask import request
    unittest.main()
