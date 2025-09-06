import asyncio
import telebot
import requests
from abc import ABC, abstractmethod
from telebot.types import Message
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.logger import logger
from RAG_BOT.src.persistence.user_settings_manager import UserSettingsManager
from RAG_BOT.src.persistence.conversation_interfaces import AbstractThreadManager
from RAG_BOT.src.services.message_processor import MessageProcessor
from urllib3.exceptions import ProtocolError
from requests.exceptions import ConnectionError, Timeout, RequestException

class BaseHandler(ABC):
    """A base class for command and message handlers to inherit common dependencies."""
    def __init__(
        self,
        bot: telebot.TeleBot,
        config: Config,
        user_settings_manager: UserSettingsManager,
        thread_manager: AbstractThreadManager,
        message_processor: MessageProcessor,
        loop: asyncio.AbstractEventLoop,
        **kwargs # To catch extra dependencies like processors
    ):
        self.bot = bot
        self.config = config
        self.user_settings_manager = user_settings_manager
        self.thread_manager = thread_manager
        self.message_processor = message_processor
        self.loop = loop
        # Store any other dependencies passed via kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


    @abstractmethod
    def handle(self, message: Message):
        """Handles an incoming message. This must be implemented by subclasses."""
        pass


    def send_response(self, message: Message, user_id: str, response_text: str):
        """Sends a response to the user, handling potential message length limits."""
        if not response_text:
            logger.warning(f"Attempted to send empty response to user {user_id}")
            response_text = "Sorry, I could not generate a response."

        max_telegram_length = 4096
        chunks = [response_text[i:i + max_telegram_length] for i in range(0, len(response_text), max_telegram_length)]
        try:
            if chunks:
                self.bot.reply_to(message, chunks[0])
                for chunk in chunks[1:]:
                    self.bot.send_message(message.chat.id, chunk)
        except Exception as e:
             logger.error(f"Unexpected error in send_response for user {user_id}: {e}", exc_info=True)
             self.bot.reply_to(message, "Sorry, there was an error sending the full response.")


    def _is_retriable_network_error(self, exception):
            """Check if an exception is a retriable network error."""
            retriable_errors = (
                ConnectionError,
                Timeout,
                ProtocolError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
            )            
            if isinstance(exception, retriable_errors):
                return True
                
            # Check for specific error messages
            error_str = str(exception).lower()
            retriable_messages = [
                "remote end closed connection",
                "connection aborted",
                "connection broken",
                "connection reset",
                "timeout",
                "timed out",
                "max retries exceeded"
            ]            
            return any(msg in error_str for msg in retriable_messages)


    async def _send_message_with_retry(self, send_func, *args, max_retries=3, base_delay=1.0):
        """
        Send a message with exponential backoff retry logic.
        
        Args:
            send_func: The function to call (bot.reply_to or bot.send_message)
            *args: Arguments to pass to send_func
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
        """
        last_exception = None        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                # Force session recreation before retry attempts
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{max_retries}, forcing session recreation...")
                    # Force new session creation by clearing the session
                    if hasattr(self.bot, 'token'):
                        import telebot.apihelper as apihelper
                        # Force session recreation
                        if hasattr(apihelper, '_session'):
                            apihelper._session = None
                        
                # Add small delay before each attempt to prevent rate limiting
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.info(f"Waiting {delay}s before retry attempt {attempt}")
                    await asyncio.sleep(delay)
                else:
                    # Small delay even on first attempt to let any pending operations complete
                    await asyncio.sleep(0.1)
                
                # Execute the actual send operation
                # self.loop is the asyncio event loop that is running in your bot's background thread.
                # run_in_executor: This method tells the event loop: "Don't run this blocking function here. Instead, 
                # run it in a separate background thread from a thread pool."
                # send_func is the synchronous, blocking function that we want to run in the separate thread.
                # args are the arguments to pass to the blocking function (send_func)
                # None means use the default ThreadPoolExecutor
                result = await self.loop.run_in_executor(None, send_func, *args)                
                if attempt > 0:
                    logger.info(f"Message sent successfully on retry attempt {attempt}")
                
                return result                
            except Exception as e:
                last_exception = e
                error_msg = str(e)
                logger.warning(f"Send attempt {attempt + 1} failed: {error_msg}")                
                # If this is not a retriable error, don't retry
                if not self._is_retriable_network_error(e):
                    logger.error(f"Non-retriable error encountered: {error_msg}")
                    break
                    
                # If this was the last attempt, don't log as retry
                if attempt < max_retries:
                    logger.info(f"Retriable error detected, will retry. Attempts remaining: {max_retries - attempt}")
        
        # All retries exhausted
        logger.error(f"All {max_retries + 1} send attempts failed. Last error: {last_exception}")
        raise last_exception


    async def send_response_async(self, message: Message, user_id: str, response_text: str):
        """
        Asynchronously sends a response to the user with robust retry logic.
        """
        if not response_text:
            logger.warning(f"Attempted to send empty async response to user {user_id}")
            response_text = "Sorry, I could not generate a response."

        max_telegram_length = 4096
        chunks = [response_text[i:i + max_telegram_length] for i in range(0, len(response_text), max_telegram_length)]
        
        try:
            if chunks:
                logger.debug(f"Sending response to user {user_id} in {len(chunks)} chunk(s)")                
                # Send first chunk as reply
                await self._send_message_with_retry(self.bot.reply_to, message, chunks[0])                
                # Send remaining chunks as regular messages
                for i, chunk in enumerate(chunks[1:], 1):
                    logger.debug(f"Sending chunk {i + 1}/{len(chunks)} to user {user_id}")
                    # Small delay between chunks to prevent rate limiting
                    await asyncio.sleep(0.2)
                    await self._send_message_with_retry(self.bot.send_message, message.chat.id, chunk)

                logger.info(f"Successfully sent {len(chunks)} chunk(s) to user {user_id}")                
        except Exception as e:
            logger.error(f"All retry attempts failed for user {user_id}: {e}", exc_info=True)            
            # Final fallback: try to send error message with single retry
            try:
                error_msg = "Sorry, there was an error sending the full response. Please try again."
                await self._send_message_with_retry(
                    self.bot.reply_to, 
                    message, 
                    error_msg, 
                    max_retries=1
                )
            except Exception as fallback_error:
                logger.error(f"Even error message failed to send to user {user_id}: {fallback_error}")
