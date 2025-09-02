import asyncio
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.logger import logger
from langchain_core.messages import HumanMessage
from telebot.types import Message
from langgraph.graph import StateGraph
from RAG_BOT.src.json_parser import JsonParser
from RAG_BOT.src.persistence.conversation_interfaces import AbstractThreadManager
from RAG_BOT.src.utils import detect_text_language

class MessageProcessor:
    """
    Processes incoming messages, manages conversation threads, and interacts with the RAG agent.
    """
    def __init__(self, agent: StateGraph, config: Config, json_parser: JsonParser, thread_manager: AbstractThreadManager):
        self.config = config
        self.agent = agent
        self.json_parser = json_parser
        self.thread_manager = thread_manager


    async def _invoke_agent_and_get_response(self, thread_id: str, language_code: str, mode: str, message: str) -> tuple[str, bool]:
        """
        Invokes the RAG agent, extracts the response, and checks if a summary was triggered.
        Returns the response content and the summary trigger flag.
        """
        final_state = None
        try:
            logger.info(f"Invoking agent for thread_id={thread_id}, lang='{language_code}', mode='{mode}', query='{message[:50]}...' ")
            config_thread = {"configurable": {"thread_id": thread_id}}
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "language_code": language_code,
                "mode": mode # Pass mode to agent state
            }
            final_state = await self.agent.ainvoke(initial_state, config_thread)            
            answer = None
            try:
                last_msg_content = final_state.get("messages", [])[-1].content
                json_result = self.json_parser.parse_json_answer(last_msg_content)
                
                if json_result and "answer" in json_result:
                    answer = json_result.get("answer")
                    # Only include references if in 'research' mode
                    if mode == 'research':
                        references = json_result.get("references")
                        if references and isinstance(references, list):
                            ref_string = "\n\n*References:*\n- " + "\n- ".join(sorted(list(set(references))))
                            answer += ref_string
            except (KeyError, IndexError, AttributeError, Exception) as e:
                logger.warning(f"Could not extract answer from agent response: {e}")
            
            if answer is None:
                answer = "Sorry, I couldn't retrieve an answer for that."

            summary_triggered = final_state.get('summary_was_triggered', False)
            return answer, summary_triggered

        except Exception as e:
            logger.error(f"Error invoking RAG agent for thread_id {thread_id}: {str(e)}", exc_info=True)
            error_message = "Sorry, I encountered an internal error."
            return error_message, False
        

    async def process_message(self, incoming_message: Message, language_code: str, mode: str):
        user_id = str(incoming_message.from_user.id)
        message_text = incoming_message.text
        if not message_text:
            logger.warning(f"Received empty message text from user {user_id}")
            return "Sorry, I didn't receive any text."

        text_language = detect_text_language(message_text, default_lang=language_code)
        if text_language != language_code:
            logger.info(f"Detected message language '{text_language}' differs from user setting '{language_code}'. Using detected language.")
            language_code = text_language

        logger.info(f"Processing message from {user_id}: {message_text[:100]}...")
        active_thread_id = await self.thread_manager.get_active_thread_id(user_id)
        if not active_thread_id:
            logger.error(f"Could not get or create an active thread for user {user_id}")
            return "I'm having trouble remembering our conversation. Please try again."

        response, summary_triggered = await self._invoke_agent_and_get_response(
            thread_id=active_thread_id,
            language_code=language_code,
            mode=mode,
            message=message_text
        )
        # After any interaction, update the thread's last modified time.
        await self.thread_manager.update_thread_last_modified(active_thread_id)
        if summary_triggered:
            logger.info(f"Summary was triggered for thread {active_thread_id} for user {user_id}. Checking threshold.")
            new_count = await self.thread_manager.increment_summary_count(user_id, active_thread_id)
            if new_count >= self.config.CONVERSATION_SUMMARY_THRESHOLD:
                logger.info(f"Threshold of {self.config.CONVERSATION_SUMMARY_THRESHOLD} met for user {user_id}. Archiving old thread and creating a new one.")
                await self.thread_manager.archive_thread(user_id, active_thread_id)
                # Create a new thread, letting the manager handle the default title to ensure timestamp consistency.
                await self.thread_manager.create_new_thread(user_id)
                notification = "\n\n*To keep our conversation focused, I've started a new thread. You can always review our past chats with the /threads command.*"
                response += notification

        logger.info(f"Generated response for user {user_id}: {response[:100]}...")
        return response
