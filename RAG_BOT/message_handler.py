# /home/bk_anupam/code/LLM_agents/RAG_BOT/message_handler.py
import datetime
from RAG_BOT.config import Config
from RAG_BOT.logger import logger # Corrected import path based on other files
from langchain_core.messages import HumanMessage
from telebot.types import Message
from langgraph.graph import StateGraph
from RAG_BOT.utils import parse_json_answer


class MessageHandler:
    def __init__(self, agent: StateGraph, config: Config):
        # Store user session data (could be moved to a database for persistence)
        self.config = config
        self.sessions = config.USER_SESSIONS
        self.agent = agent

    def _get_user_session(self, user_id):
        if user_id not in self.sessions:
            logger.info(f"Creating new session for user {user_id}")
            self.sessions[user_id] = {
                'last_interaction': datetime.datetime.now(),
                'conversation': [],
                'context': {},
                'language': 'en',  # Default language
                'mode': 'default'  # Default mode
            }
        # Ensure essential keys are always present using setdefault
        self.sessions[user_id].setdefault('language', 'en')
        self.sessions[user_id].setdefault('mode', 'default')
        self.sessions[user_id].setdefault('conversation', [])
        return self.sessions[user_id]

    def _update_session(self, user_id, message, response):
        """Update the user session with new interaction"""
        session = self._get_user_session(user_id)
        session['last_interaction'] = datetime.datetime.now()
        session['conversation'].append({
            'user': message,
            'bot': response,
            'timestamp': datetime.datetime.now().isoformat()
        })
        # Limit conversation history (optional)
        history_limit = self.config.CONVERSATION_HISTORY_LIMIT or 10
        if len(session['conversation']) > history_limit:
            session['conversation'] = session['conversation'][-history_limit:]


    async def _invoke_agent_and_get_response(self, chat_id: int, language_code: str, mode: str, message: str) -> str:
        """
        Invokes the RAG agent and extracts the response.
        Handles potential errors and unexpected response formats.
        """
        try:
            logger.info(f"Invoking agent for thread_id={str(chat_id)}, lang='{language_code}', mode='{mode}', query='{message[:50]}...'")
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
        session = self._get_user_session(user_id)
        message_lower = message.lower().strip()
        
        greetings = {'hello', 'hi', 'hey'}
        if message_lower in greetings:
             response = "👋 Hello! I'm your Telegram assistant. How can I help you today?"
        elif "help" == message_lower:
            response = ("Here's what I can do:\n"
                    "- Answer your query (on pdf documents uploaded). Use /query command followed by the query for this\n"
                    "- Index and store in vector DB uploaded pdf documents. Just send the pdf document as a message\n"
                    "- Answer any general query \n"
                    "- Last message - to see your last message\n"
                    "Just let me know what you need!")
        elif "last message" == message_lower:
            if session['conversation']:
                last_user_message = session['conversation'][-1]['user']
                response = f"Your last message was: '{last_user_message}'"
            else:
                response = "You haven't sent any previous messages in this session."
        else:
            response = await self._invoke_agent_and_get_response(
                chat_id=incoming_message.chat.id,
                language_code=language_code,
                mode=mode, # Pass mode to the agent invocation
                message=message
            )

        self._update_session(user_id, message, response)
        logger.info(f"Generated response for user {user_id}: {response[:100]}...")
        return response
