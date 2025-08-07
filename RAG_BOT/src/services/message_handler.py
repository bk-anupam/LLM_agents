import datetime
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.logger import logger
from langchain_core.messages import HumanMessage
from telebot.types import Message
from langgraph.graph import StateGraph
from RAG_BOT.src.utils import parse_json_answer


class MessageHandler:
    def __init__(self, agent: StateGraph, config: Config):
        self.config = config
        self.agent = agent

    async def _invoke_agent_and_get_response(self, chat_id: int, language_code: str, mode: str, message: str) -> str:
        """
        Invokes the RAG agent and extracts the response.
        Handles potential errors and unexpected response formats.
        """
        try:
            logger.info(f"Invoking agent for thread_id={str(chat_id)}, lang='{language_code}', mode='{mode}', query='{message[:50]}...' ")
            config_thread = {"configurable": {"thread_id": str(chat_id)}}
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "language_code": language_code,
                "mode": mode # Pass mode to agent state
            }
            final_state = await self.agent.ainvoke(initial_state, config_thread)
            answer = None
            try:
                last_msg_content = final_state.get("messages", [])[-1].content
                json_result = parse_json_answer(last_msg_content)
                
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
            return answer
        except Exception as e:
            logger.error(f"Error invoking RAG agent for chat_id {chat_id}: {str(e)}", exc_info=True)
            return "Sorry, I encountered an internal error while processing your query."


    async def process_message(self, incoming_message: Message, language_code: str, mode: str):
        """
        Process the incoming message and generate a response
        This is where you implement your custom logic
        """
        user_id = incoming_message.from_user.id
        message = incoming_message.text
        if not message:
             logger.warning(f"Received empty message text from user {user_id}")
             return "Sorry, I didn't receive any text."

        logger.info(f"Processing message from {user_id}: {message[:100]}...")
        message_lower = message.lower().strip()
        
        greetings = {'hello', 'hi', 'hey'}
        if message_lower in greetings:
             response = "👋 Hello! I'm your Telegram assistant. How can I help you today?"
        elif "help" == message_lower:
            response = ("Here's what I can do:\n"
                    "- Answer your query (on pdf documents uploaded). Use /query command followed by the query for this\n"
                    "- Index and store in vector DB uploaded pdf documents. Just send the pdf document as a message\n"
                    "- Answer any general query \n"
                    "Just let me know what you need!")
        else:
            response = await self._invoke_agent_and_get_response(
                chat_id=incoming_message.chat.id,
                language_code=language_code,
                mode=mode, # Pass mode to the agent invocation
                message=message
            )

        logger.info(f"Generated response for user {user_id}: {response[:100]}...")
        return response
