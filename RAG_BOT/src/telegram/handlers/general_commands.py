from telebot.types import Message
from RAG_BOT.src.logger import logger
from RAG_BOT.src.telegram.handlers.base_handler import BaseHandler

class StartCommand(BaseHandler):
    """Handles the /start command."""

    def handle(self, message: Message):
        logger.info(f"Received /start command from user {message.from_user.id}")
        self.bot.reply_to(message, "Welcome to the spiritual chatbot! Ask me questions about the indexed documents, or use /help for commands.")


class HelpCommand(BaseHandler):
    """Handles the /help command."""

    def handle(self, message: Message):
        logger.info(f"Received /help command from user {message.from_user.id}")
        self.bot.reply_to(message,
            """
            Available Commands:
            /start - Show welcome message.
            /help - Show this help message.
            /language <lang> - Set bot language (english or hindi). Example: /language hindi
            /mode <mode> - Set bot response mode (default or research). Example: /mode research
            /new - Start a new conversation thread.
            /threads - List your recent conversations.
            /switch <number> - Switch to a different conversation thread.
            /delete <number> - Delete a conversation thread.
            You can also just type your question directly.
            """
        )
