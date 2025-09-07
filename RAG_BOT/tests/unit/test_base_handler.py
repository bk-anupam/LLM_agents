import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, call
from telebot.types import Message
import requests
from urllib3.exceptions import ProtocolError

# The class to be tested
from RAG_BOT.src.telegram.handlers.base_handler import BaseHandler

# A concrete implementation for testing
class ConcreteHandler(BaseHandler):
    def handle(self, message: Message):
        pass # Not needed for testing the base class methods

class TestBaseHandler(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_bot = MagicMock()
        self.mock_config = MagicMock()
        self.mock_user_settings_manager = MagicMock()
        self.mock_thread_manager = MagicMock()
        self.mock_message_processor = MagicMock()
        self.loop = asyncio.get_event_loop()

        # Instantiate the concrete handler
        self.handler = ConcreteHandler(
            bot=self.mock_bot,
            config=self.mock_config,
            user_settings_manager=self.mock_user_settings_manager,
            thread_manager=self.mock_thread_manager,
            message_processor=self.mock_message_processor,
            loop=self.loop,
            extra_dep="some_value" # Test kwargs
        )

        # Create a dummy message object
        self.dummy_message = MagicMock(spec=Message)
        self.dummy_message.from_user = MagicMock()
        self.dummy_message.chat = MagicMock()
        self.dummy_message.from_user.id = 123
        self.dummy_message.chat.id = 456

    def test_init_with_kwargs(self):
        """Test that extra dependencies are set as attributes."""
        self.assertTrue(hasattr(self.handler, 'extra_dep'))
        self.assertEqual(self.handler.extra_dep, "some_value")

    # --- Tests for send_response (synchronous) ---

    def test_send_response_short_message(self):
        """Test sending a short response."""
        self.handler.send_response(self.dummy_message, "123", "Hello")
        self.mock_bot.reply_to.assert_called_once_with(self.dummy_message, "Hello")
        self.mock_bot.send_message.assert_not_called()

    def test_send_response_long_message(self):
        """Test sending a response that needs to be chunked."""
        long_text = "A" * 5000
        chunk1 = "A" * 4096
        chunk2 = "A" * (5000 - 4096)

        self.handler.send_response(self.dummy_message, "123", long_text)

        self.mock_bot.reply_to.assert_called_once_with(self.dummy_message, chunk1)
        self.mock_bot.send_message.assert_called_once_with(self.dummy_message.chat.id, chunk2)

    def test_send_response_empty_message(self):
        """Test sending an empty response results in a fallback message."""
        self.handler.send_response(self.dummy_message, "123", "")
        self.mock_bot.reply_to.assert_called_once_with(self.dummy_message, "Sorry, I could not generate a response.")
        self.mock_bot.send_message.assert_not_called()

    @patch('RAG_BOT.src.telegram.handlers.base_handler.logger')
    def test_send_response_exception(self, mock_logger):
        """Test that an exception during sending is caught and logged."""
        # First call raises an error, second call (the fallback) succeeds
        self.mock_bot.reply_to.side_effect = [Exception("Test error"), None]
        
        self.handler.send_response(self.dummy_message, "123", "Hello")

        # Check that reply_to was called twice
        self.assertEqual(self.mock_bot.reply_to.call_count, 2)
        
        # Check the arguments of both calls
        calls = [
            call(self.dummy_message, "Hello"),
            call(self.dummy_message, "Sorry, there was an error sending the full response.")
        ]
        self.mock_bot.reply_to.assert_has_calls(calls)
        
        # Check that the error was logged
        mock_logger.error.assert_called()

    # --- Tests for _is_retriable_network_error ---

    def test_is_retriable_network_error_true_by_type(self):
        """Test retriable errors by their type."""
        retriable_exceptions = [
            requests.exceptions.ConnectionError(),
            requests.exceptions.Timeout(),
            ProtocolError(),
            requests.exceptions.ChunkedEncodingError(),
        ]
        for exc in retriable_exceptions:
            with self.subTest(exception=exc):
                self.assertTrue(self.handler._is_retriable_network_error(exc))

    def test_is_retriable_network_error_true_by_message(self):
        """Test retriable errors by their string message."""
        retriable_messages = [
            "remote end closed connection without response",
            "connection aborted",
            "connection broken",
            "connection reset by peer",
            "read timed out",
            "max retries exceeded with url",
        ]
        for msg in retriable_messages:
            with self.subTest(message=msg):
                exc = Exception(msg)
                self.assertTrue(self.handler._is_retriable_network_error(exc))

    def test_is_retriable_network_error_false(self):
        """Test non-retriable errors."""
        non_retriable_exceptions = [
            ValueError("Some value error"),
            TypeError("Some type error"),
            Exception("Some generic error"),
        ]
        for exc in non_retriable_exceptions:
            with self.subTest(exception=exc):
                self.assertFalse(self.handler._is_retriable_network_error(exc))

    # --- Tests for _send_message_with_retry (asynchronous) ---

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_send_with_retry_success_first_try(self, mock_sleep):
        """Test successful send on the first attempt."""
        send_func = MagicMock(return_value="Success")
        self.handler.loop.run_in_executor = AsyncMock(return_value="Success")

        result = await self.handler._send_message_with_retry(send_func, "arg1", "arg2")

        self.assertEqual(result, "Success")
        self.handler.loop.run_in_executor.assert_called_once_with(None, send_func, "arg1", "arg2")
        # sleep is called once at the beginning
        self.assertEqual(mock_sleep.call_count, 1)


    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_send_with_retry_success_on_retry(self, mock_sleep):
        """Test successful send after one failed attempt."""
        send_func = MagicMock()
        self.handler.loop.run_in_executor = AsyncMock()
        self.handler.loop.run_in_executor.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            "Success"
        ]

        result = await self.handler._send_message_with_retry(send_func, "arg1", max_retries=2)

        self.assertEqual(result, "Success")
        self.assertEqual(self.handler.loop.run_in_executor.call_count, 2)
        # sleep is called at the beginning, and then for the retry
        self.assertEqual(mock_sleep.call_count, 2)

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_send_with_retry_all_fails(self, mock_sleep):
        """Test that an exception is raised after all retries fail."""
        send_func = MagicMock()
        self.handler.loop.run_in_executor = AsyncMock()
        last_exception = requests.exceptions.Timeout("Timed out")
        self.handler.loop.run_in_executor.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            last_exception
        ]

        with self.assertRaises(requests.exceptions.Timeout):
            await self.handler._send_message_with_retry(send_func, "arg1", max_retries=1)

        self.assertEqual(self.handler.loop.run_in_executor.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_send_with_retry_non_retriable_error(self, mock_sleep):
        """Test that it does not retry on a non-retriable error."""
        send_func = MagicMock()
        self.handler.loop.run_in_executor = AsyncMock()
        non_retriable_exception = ValueError("Invalid value")
        self.handler.loop.run_in_executor.side_effect = non_retriable_exception

        with self.assertRaises(ValueError):
            await self.handler._send_message_with_retry(send_func, "arg1", max_retries=3)

        self.handler.loop.run_in_executor.assert_called_once()
        # sleep is called once at the beginning
        self.assertEqual(mock_sleep.call_count, 1)

    # --- Tests for send_response_async (asynchronous) ---

    async def test_send_response_async_short_message(self):
        """Test sending a short async response."""
        self.handler._send_message_with_retry = AsyncMock()
        await self.handler.send_response_async(self.dummy_message, "123", "Hello")

        self.handler._send_message_with_retry.assert_called_once_with(
            self.mock_bot.reply_to, self.dummy_message, "Hello"
        )

    async def test_send_response_async_long_message(self):
        """Test sending a long async response that needs chunking."""
        self.handler._send_message_with_retry = AsyncMock()
        long_text = "A" * 5000
        chunk1 = "A" * 4096
        chunk2 = "A" * (5000 - 4096)

        await self.handler.send_response_async(self.dummy_message, "123", long_text)

        calls = [
            call(self.mock_bot.reply_to, self.dummy_message, chunk1),
            call(self.mock_bot.send_message, self.dummy_message.chat.id, chunk2)
        ]
        self.handler._send_message_with_retry.assert_has_calls(calls)
        self.assertEqual(self.handler._send_message_with_retry.call_count, 2)

    async def test_send_response_async_empty_message(self):
        """Test sending an empty async response results in a fallback."""
        self.handler._send_message_with_retry = AsyncMock()
        await self.handler.send_response_async(self.dummy_message, "123", "")

        self.handler._send_message_with_retry.assert_called_once_with(
            self.mock_bot.reply_to, self.dummy_message, "Sorry, I could not generate a response."
        )

    @patch('RAG_BOT.src.telegram.handlers.base_handler.logger')
    async def test_send_response_async_fails_and_sends_fallback(self, mock_logger):
        """Test that a fallback message is sent if the main send fails."""
        self.handler._send_message_with_retry = AsyncMock()
        main_error = Exception("Main send failed")
        self.handler._send_message_with_retry.side_effect = [
            main_error, # First call for the main message fails
            "Success"   # Second call for the fallback message succeeds
        ]

        await self.handler.send_response_async(self.dummy_message, "123", "Hello")

        self.assertEqual(self.handler._send_message_with_retry.call_count, 2)
        # First call fails
        self.handler._send_message_with_retry.assert_any_call(
            self.mock_bot.reply_to, self.dummy_message, "Hello"
        )
        # Second call is the fallback
        self.handler._send_message_with_retry.assert_called_with(
            self.mock_bot.reply_to,
            self.dummy_message,
            "Sorry, there was an error sending the full response. Please try again.",
            max_retries=1
        )
        mock_logger.error.assert_called_with(f"All retry attempts failed for user 123: {main_error}", exc_info=True)

    @patch('RAG_BOT.src.telegram.handlers.base_handler.logger')
    async def test_send_response_async_fallback_also_fails(self, mock_logger):
        """Test that an error is logged if the fallback message also fails."""
        self.handler._send_message_with_retry = AsyncMock()
        main_error = Exception("Main send failed")
        fallback_error = Exception("Fallback send failed")
        self.handler._send_message_with_retry.side_effect = [main_error, fallback_error]

        await self.handler.send_response_async(self.dummy_message, "123", "Hello")

        self.assertEqual(self.handler._send_message_with_retry.call_count, 2)
        mock_logger.error.assert_any_call(f"All retry attempts failed for user 123: {main_error}", exc_info=True)
        mock_logger.error.assert_called_with(f"Even error message failed to send to user 123: {fallback_error}")

if __name__ == '__main__':
    unittest.main()
