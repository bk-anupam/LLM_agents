import asyncio
from telebot.types import Message
from RAG_BOT.src.logger import logger
from .base_handler import BaseHandler

class LanguageCommand(BaseHandler):
    """Handles the /language command to set user preference."""

    def handle(self, message: Message):
        user_id = message.from_user.id
        parts = message.text.split(maxsplit=1)

        if len(parts) < 2:
            usage_text = self.config.get_user_message('language_usage_help', "Usage: /language <english|hindi>")
            self.bot.reply_to(message, usage_text)
            return

        lang_input = parts[1].strip().lower()
        lang_code = {'english': 'en', 'hindi': 'hi'}.get(lang_input)

        if not lang_code:
            unsupported_text = self.config.get_user_message('language_unsupported', "Unsupported language. Please use 'english' or 'hindi'.")
            self.bot.reply_to(message, unsupported_text)
            return

        future = asyncio.run_coroutine_threadsafe(
            self.user_settings_manager.update_user_settings(user_id, language_code=lang_code),
            self.loop
        )
        try:
            future.result(timeout=10)
            logger.info(f"Successfully set language for user {user_id} to '{lang_code}'")
            confirmation_key = f"language_set_{lang_code}"
            defaults = {'en': "Language set to English.", 'hi': "भाषा हिंदी में सेट कर दी गई है।"}
            reply_text = self.config.get_user_message(confirmation_key, defaults[lang_code])
            self.bot.reply_to(message, reply_text)
        except Exception as e:
            logger.error(f"Failed to update language settings for user {user_id}: {e}", exc_info=True)
            self.bot.reply_to(message, "Sorry, there was an error saving your preference.")