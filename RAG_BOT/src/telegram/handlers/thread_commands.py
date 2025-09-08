import asyncio
import datetime
from telebot.types import Message
from RAG_BOT.src.logger import logger
from RAG_BOT.src.telegram.handlers.base_handler import BaseHandler

class NewThreadCommand(BaseHandler):
    """Handles the /new command to create a new conversation thread."""

    def handle(self, message: Message):
        """Synchronous wrapper for the async new command handler."""
        asyncio.run_coroutine_threadsafe(self._handle_async(message), self.loop)

    async def _handle_async(self, message: Message):
        user_id = str(message.from_user.id)
        logger.info(f"Received /new command from user {user_id}")
        try:
            # Allow creating a thread with a custom title, e.g., /new My Project Chat
            parts = message.text.split(maxsplit=1)
            title = parts[1] if len(parts) > 1 else None

            # Let the thread manager handle default title generation if none is provided.
            await self.thread_manager.create_new_thread(user_id, title=title)
            reply_text = self.config.get_user_message('new_thread_manual', "Starting a new conversation thread. Your next message will begin a new chat.")
            await self.send_response_async(message, user_id, reply_text)

        except Exception as e:
            logger.error(f"Error handling /new command for user {user_id}: {e}", exc_info=True)
            error_message = self.config.get_user_message('error_new_thread', "Sorry, I couldn't start a new conversation. Please try again.")
            await self.send_response_async(message, user_id, error_message)
 

class ListThreadsCommand(BaseHandler):
    """Handles the /threads command to list conversation threads."""

    def handle(self, message: Message):
        """Synchronous wrapper for the async threads command handler."""
        asyncio.run_coroutine_threadsafe(self._handle_async(message), self.loop)

    async def _handle_async(self, message: Message):
        user_id = str(message.from_user.id)
        logger.info(f"Received /threads command from user {user_id}")
        try:
            threads = await self.thread_manager.list_threads(user_id)
            logger.info(f"Found {len(threads)} threads for user {user_id}")
            if not threads:
                no_threads_msg = self.config.get_user_message('threads_none', "You have no conversation threads yet.")
                await self.send_response_async(message, user_id, no_threads_msg)
                return

            response_lines = [self.config.get_user_message('threads_header', "Your recent conversations:")]
            for i, thread in enumerate(threads[:10], 1): # Show top 10
                status = "[ACTIVE]" if thread.is_active else "[Archived]" if thread.is_archived else ""
                if thread.last_modified_at and isinstance(thread.last_modified_at, datetime.datetime):
                    # Firestore returns UTC-aware datetime objects. We format it and explicitly add the timezone.
                    last_modified_str = thread.last_modified_at.strftime('%Y-%m-%d %H:%M') + " UTC"
                else:
                    last_modified_str = "N/A"

                response_lines.append(f"{i}. {status} {thread.title} (last active: {last_modified_str})")

            response_lines.append(f"\n{self.config.get_user_message('threads_footer', 'To switch, use /switch <number>. To delete, use /delete <number>.')}")
            await self.send_response_async(message, user_id, "\n".join(response_lines))
        except Exception as e:
            logger.error(f"Error handling /threads command for user {user_id}: {e}", exc_info=True)
            error_message = self.config.get_user_message('error_list_threads', "Sorry, I couldn't retrieve your conversation list.")
            await self.send_response_async(message, user_id, error_message)


class SwitchThreadCommand(BaseHandler):
    """Handles the /switch command to change the active thread."""

    def handle(self, message: Message):
        """Synchronous wrapper for the async switch command handler."""
        asyncio.run_coroutine_threadsafe(self._handle_async(message), self.loop)

    async def _handle_async(self, message: Message):
        user_id = str(message.from_user.id)
        logger.info(f"Received /switch command from user {user_id}")
        try:
            parts = message.text.split()
            if len(parts) < 2 or not parts[1].isdigit():
                usage_msg = self.config.get_user_message('switch_usage', "Usage: /switch <thread_number>")
                await self.send_response_async(message, user_id, usage_msg)
                return

            thread_number = int(parts[1])
            threads = await self.thread_manager.list_threads(user_id)

            if not 1 <= thread_number <= len(threads):
                invalid_num_msg = self.config.get_user_message('switch_invalid_number', "Invalid thread number.")
                await self.send_response_async(message, user_id, invalid_num_msg)
                return

            target_thread = threads[thread_number - 1]
            if target_thread.is_active:
                already_active_msg = f"You are already in the conversation: '{target_thread.title}'."
                await self.send_response_async(message, user_id, already_active_msg)
                return

            await self.thread_manager.switch_active_thread(user_id, target_thread.thread_id)
            reply_text = self.config.get_user_message('switch_success', "Switched to conversation: '{title}'").format(title=target_thread.title)
            await self.send_response_async(message, user_id, reply_text)
        except Exception as e:
            logger.error(f"Error handling /switch command for user {user_id}: {e}", exc_info=True)
            error_message = self.config.get_user_message('error_switch_thread', "Sorry, I couldn't switch conversations.")
            await self.send_response_async(message, user_id, error_message)


class DeleteThreadCommand(BaseHandler):
    """Handles the /delete command to delete a conversation thread."""

    def handle(self, message: Message):
        """Synchronous wrapper for the async delete command handler."""
        asyncio.run_coroutine_threadsafe(self._handle_async(message), self.loop)

    async def _handle_async(self, message: Message):
        user_id = str(message.from_user.id)
        logger.info(f"Received /delete command from user {user_id}")
        try:
            parts = message.text.split()
            if len(parts) < 2 or not parts[1].isdigit():
                usage_msg = self.config.get_user_message('delete_usage', "Usage: /delete <thread_number>")
                await self.send_response_async(message, user_id, usage_msg)
                return

            thread_number = int(parts[1])
            threads = await self.thread_manager.list_threads(user_id)

            if not 1 <= thread_number <= len(threads):
                invalid_num_msg = self.config.get_user_message('delete_invalid_number', "Invalid thread number.")
                await self.send_response_async(message, user_id, invalid_num_msg)
                return

            target_thread = threads[thread_number - 1]
            if target_thread.is_active:
                active_thread_msg = self.config.get_user_message('delete_active_thread', "You cannot delete the active conversation. Please /switch to another thread first.")
                await self.send_response_async(message, user_id, active_thread_msg)
                return

            deleted = await self.thread_manager.delete_thread(user_id, target_thread.thread_id)

            if deleted:
                reply_text = self.config.get_user_message('delete_success', "Deleted conversation: '{title}'").format(title=target_thread.title)
                await self.send_response_async(message, user_id, reply_text)
            else:
                raise Exception("Thread manager failed to delete the thread.")

        except Exception as e:
            logger.error(f"Error handling /delete command for user {user_id}: {e}", exc_info=True)
            error_message = "Sorry, I couldn't delete that conversation."
            await self.send_response_async(message, user_id, error_message)
