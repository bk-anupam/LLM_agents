import asyncio
from telebot.types import Message
from RAG_BOT.src.logger import logger
from .base_handler import BaseHandler

class TextMessageHandler(BaseHandler):
    """Handles all general text messages that are not commands."""

    def handle(self, message: Message):
        """
        Schedules the asynchronous message processing to run in the background
        event loop without blocking the synchronous webhook response. This is
        critical for long-running agent tasks.
        """
        asyncio.run_coroutine_threadsafe(self._handle_and_reply_async(message), self.loop)


    async def _handle_and_reply_async(self, message: Message):
        """
        The core async method that processes the message and sends the reply.
        This runs entirely in the background thread's event loop.
        """
        user_id = message.from_user.id
        response_text = "Sorry, an unexpected error occurred."
        try:
            if not self.message_processor:
                logger.error(f"MessageProcessor not initialized. Cannot process message for user {user_id}.")
                response_text = "The bot is currently initializing. Please try again shortly."
            else:
                response_text = await asyncio.wait_for(
                    self._handle_async_core(message),
                    timeout=self.config.ASYNC_OPERATION_TIMEOUT
                )
        except asyncio.TimeoutError:
            logger.error(f"Async message processing timed out for user {user_id}")
            response_text = "Sorry, your request timed out. Please try again."
        except Exception as e:
            logger.error(f"Error during async message processing for user {user_id}: {e}", exc_info=True)
        finally:
            # Use the async version of send_response
            await self.send_response_async(message, user_id, response_text)


    async def _handle_async_core(self, message: Message) -> str:
        """Core async logic for handling messages. Runs in the dedicated event loop."""
        user_id = message.from_user.id
        settings = await self.user_settings_manager.get_user_settings(user_id)
        user_lang = settings.get('language_code', 'en')
        user_mode = settings.get('mode', 'default')
        logger.info(f"Processing message for user {user_id} in async core: '{message.text[:100]}...' (Lang: {user_lang}, Mode: {user_mode})")
        try:
            return await self.message_processor.process_message(message, user_lang, user_mode)
        except Exception as e:
            logger.error(f"Error in _handle_all_messages_async_core for user {user_id}: {str(e)}", exc_info=True)
            return "Sorry, I encountered an internal error while processing your request."
