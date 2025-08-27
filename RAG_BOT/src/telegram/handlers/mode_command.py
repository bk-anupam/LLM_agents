import asyncio
from telebot.types import Message
from RAG_BOT.src.logger import logger
from .base_handler import BaseHandler

class ModeCommand(BaseHandler):
    """Handles the /mode command to set user response mode."""

    def handle(self, message: Message):
        user_id = message.from_user.id
        parts = message.text.split(maxsplit=1)

        async def get_and_reply_current_mode():
            settings = await self.user_settings_manager.get_user_settings(user_id)
            self.bot.reply_to(message, f"Current mode is '{settings['mode']}'. Usage: /mode <default|research>")

        if len(parts) < 2:
            asyncio.run_coroutine_threadsafe(get_and_reply_current_mode(), self.loop)
            return

        new_mode = parts[1].strip().lower()
        if new_mode not in ['default', 'research']:
            self.bot.reply_to(message, "Invalid mode. Please use 'default' or 'research'.")
            return

        future = asyncio.run_coroutine_threadsafe(
            self.user_settings_manager.update_user_settings(user_id, mode=new_mode),
            self.loop
        )
        try:
            future.result(timeout=10)
            logger.info(f"Set mode preference for user {user_id} to '{new_mode}'")
            self.bot.reply_to(message, f"Mode set to '{new_mode}'.")
        except Exception as e:
            logger.error(f"Failed to update mode settings for user {user_id}: {e}", exc_info=True)
            self.bot.reply_to(message, "Sorry, there was an error saving your preference.")
