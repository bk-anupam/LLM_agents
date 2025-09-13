import asyncio
import telebot
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.logger import logger
from RAG_BOT.src.persistence.user_settings_manager import UserSettingsManager
from RAG_BOT.src.persistence.conversation_interfaces import AbstractThreadManager
from RAG_BOT.src.services.message_processor import MessageProcessor
from RAG_BOT.src.services.gcs_uploader import GCSUploaderService
from RAG_BOT.src.persistence.vector_store import VectorStore
from RAG_BOT.src.processing.pdf_processor import PdfProcessor
from RAG_BOT.src.processing.htm_processor import HtmProcessor

# Import all the new handler classes
from RAG_BOT.src.telegram.handlers.general_commands import StartCommand
from RAG_BOT.src.telegram.handlers.general_commands import HelpCommand
from RAG_BOT.src.telegram.handlers.language_command import LanguageCommand
from RAG_BOT.src.telegram.handlers.mode_command import ModeCommand
from RAG_BOT.src.telegram.handlers.thread_commands import NewThreadCommand, ListThreadsCommand, SwitchThreadCommand, DeleteThreadCommand
from RAG_BOT.src.telegram.handlers.document_handler import DocumentHandler
from RAG_BOT.src.telegram.handlers.text_message_handler import TextMessageHandler


class HandlerRegistry:
    """
    Instantiates and registers all command and message handlers for the bot.
    This class acts as a central point for managing the bot's interactive logic,
    delegating the actual work to specific handler classes.
    """
    def __init__(
        self,
        bot: telebot.TeleBot,
        config: Config,
        user_settings_manager: UserSettingsManager,
        thread_manager: AbstractThreadManager,
        message_processor: MessageProcessor,
        vector_store_instance: VectorStore,
        pdf_processor: PdfProcessor,
        htm_processor: HtmProcessor,
        gcs_uploader: GCSUploaderService,
        loop: asyncio.AbstractEventLoop,
        project_root_dir: str
    ):        
        # Create a shared dictionary of dependencies to pass to each handler
        dependencies = {
            "bot": bot,
            "config": config,
            "user_settings_manager": user_settings_manager,
            "thread_manager": thread_manager,
            "message_processor": message_processor,
            "gcs_uploader": gcs_uploader,
            "vector_store_instance": vector_store_instance,
            "pdf_processor": pdf_processor,
            "htm_processor": htm_processor,
            "loop": loop,
            "project_root_dir": project_root_dir
        }

        # Instantiate all handler classes
        self.start_command = StartCommand(**dependencies)
        self.help_command = HelpCommand(**dependencies)
        self.language_command = LanguageCommand(**dependencies)
        self.mode_command = ModeCommand(**dependencies)
        self.new_thread_command = NewThreadCommand(**dependencies)
        self.list_threads_command = ListThreadsCommand(**dependencies)
        self.switch_thread_command = SwitchThreadCommand(**dependencies)
        self.delete_thread_command = DeleteThreadCommand(**dependencies)
        self.document_handler = DocumentHandler(**dependencies)
        self.text_message_handler = TextMessageHandler(**dependencies)

        # Keep a reference to the bot for registration
        self.bot = bot


    def register_handlers(self):
        """Registers all message and command handlers with the bot instance."""
        self.bot.register_message_handler(self.start_command.handle, commands=['start'])
        self.bot.register_message_handler(self.help_command.handle, commands=['help'])
        self.bot.register_message_handler(self.language_command.handle, commands=['language'])
        self.bot.register_message_handler(self.mode_command.handle, commands=['mode'])
        self.bot.register_message_handler(self.document_handler.handle, content_types=['document'])

        # Thread management commands
        self.bot.register_message_handler(self.new_thread_command.handle, commands=['new'])
        self.bot.register_message_handler(self.list_threads_command.handle, commands=['threads'])
        self.bot.register_message_handler(self.switch_thread_command.handle, commands=['switch'])
        self.bot.register_message_handler(self.delete_thread_command.handle, commands=['delete'])

        # Catch-all handler for text messages
        self.bot.register_message_handler(self.text_message_handler.handle, func=lambda message: True, content_types=['text'])
        logger.info("Command and message handlers registered successfully.")