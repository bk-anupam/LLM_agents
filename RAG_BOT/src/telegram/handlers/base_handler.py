import asyncio
import telebot
from abc import ABC, abstractmethod
from telebot.types import Message
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.logger import logger
from RAG_BOT.src.persistence.user_settings_manager import UserSettingsManager
from RAG_BOT.src.persistence.conversation_interfaces import AbstractThreadManager
from RAG_BOT.src.services.message_processor import MessageProcessor

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


    async def send_response_async(self, message: Message, user_id: str, response_text: str):
        """
        Asynchronously sends a response to the user, handling message length limits.
        Uses run_in_executor to avoid blocking the event loop with synchronous network calls.
        """
        if not response_text:
            logger.warning(f"Attempted to send empty async response to user {user_id}")
            response_text = "Sorry, I could not generate a response."

        max_telegram_length = 4096
        chunks = [response_text[i:i + max_telegram_length] for i in range(0, len(response_text), max_telegram_length)]
        try:
            if chunks:
                await (
                    # the asyncio event loop that is running in your bot's background thread.
                    self.loop   
                        # This method tells the event loop: "Don't run this blocking function here. Instead, 
                        # run it in a separate background thread from a thread pool."
                        .run_in_executor(
                            # use the default ThreadPoolExecutor
                            None, 
                            # This is the synchronous, blocking function that we want to run in the separate thread.
                            self.bot.reply_to, 
                            # arguments to pass to the blocking function (reply_to)
                            message, 
                            chunks[0]
                        )
                )
                for chunk in chunks[1:]:
                    await self.loop.run_in_executor(None, self.bot.send_message, message.chat.id, chunk)
        except Exception as e:
            logger.error(f"Unexpected error in send_response_async for user {user_id}: {e}", exc_info=True)
            await self.loop.run_in_executor(None, self.bot.reply_to, message, "Sorry, there was an error sending the full response.")
