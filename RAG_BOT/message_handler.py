# /home/bk_anupam/code/LLM_agents/RAG_BOT/message_handler.py
import datetime
from config import Config
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
                'language': 'en'  # Default language
            }
        # Ensure essential keys are always present using setdefault
        if 'language' not in self.sessions[user_id]:
            self.sessions[user_id]['language'] = 'en'
        # *** Ensure 'conversation' key exists ***
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
        # Consider making the limit configurable
        history_limit = self.config.CONVERSATION_HISTORY_LIMIT or 10
        if len(session['conversation']) > history_limit:
            session['conversation'] = session['conversation'][-history_limit:]


    def process_message(self, incoming_message: Message, language_code: str):
        """
        Process the incoming message and generate a response
        This is where you implement your custom logic
        """
        # language_code is now passed as an argument
        user_id = incoming_message.from_user.id
        message = incoming_message.text
        if not message: # Handle cases like stickers or empty messages if needed
             logger.warning(f"Received empty message text from user {user_id}")
             return "Sorry, I didn't receive any text."

        logger.info(f"Processing message from {user_id}: {message[:100]}...") # Log snippet
        # Get user session (still useful for history, etc.)
        session = self._get_user_session(user_id)
        # Convert message to lowercase for easier matching
        message_lower = message.lower().strip()

        # --- Updated Greeting Check ---
        # Check for exact matches for greetings
        greetings = {'hello', 'hi', 'hey'}
        if message_lower in greetings:
             response = "👋 Hello! I'm your Telegram assistant. How can I help you today?"
        # --- End Updated Check ---

        elif "help" == message_lower: # Check for exact match for help too
            response = ("Here's what I can do:\n"
                    "- Answer your query (on pdf documents uploaded). Use /query command followed by the query for this\n"
                    "- Index and store in vector DB uploaded pdf documents. Just send the pdf document as a message\n"
                    "- Answer any general query \n"
                    "- Last message - to see your last message\n"
                    "Just let me know what you need!")
        elif "last message" == message_lower: # Exact match
            if session['conversation']: # Check if list is not empty
                last_user_message = session['conversation'][-1]['user']
                response = f"Your last message was: '{last_user_message}'"
            else:
                response = "You haven't sent any previous messages in this session."
        else:
            # --- Agent Invocation Logic ---
            try:
                logger.info(f"Invoking agent for thread_id={str(incoming_message.chat.id)}, lang='{language_code}', query='{message[:50]}...'")
                config_thread = {"configurable": {"thread_id": str(incoming_message.chat.id)}}
                # Build the initial state for the agent
                initial_state = {
                    "messages": [HumanMessage(content=message)],
                    "language_code": language_code
                }

                # It's good practice to stream or use async invoke if available and the agent call might take time
                # Using synchronous invoke for now as per the original code
                final_state = self.agent.invoke(initial_state, config_thread)

                # Extract the answer more robustly
                answer = None
                if isinstance(final_state, dict) and "messages" in final_state:
                    final_messages = final_state["messages"]
                    if isinstance(final_messages, list) and final_messages:
                        # Get the last message, assuming it's the agent's response
                        last_msg = final_messages[-1]
                        if hasattr(last_msg, 'content'):
                            json_result = parse_json_answer(last_msg.content)
                            answer = json_result.get("answer") if json_result else None
                        else:
                            logger.warning(f"Last message in agent response has no 'content' attribute: {last_msg}")
                    else:
                         logger.warning(f"Agent response 'messages' is not a non-empty list: {final_messages}")
                else:
                     logger.warning(f"Unexpected agent response format: {final_state}")

                if not answer:
                    answer = "Sorry, I couldn't retrieve an answer for that." # More specific default
                response = answer

            except Exception as e:
                logger.error(f"Error invoking RAG agent for user {user_id}: {str(e)}", exc_info=True)
                response = "Sorry, I encountered an internal error while processing your query."
            # --- End Agent Invocation ---

        # Update session with this interaction
        self._update_session(user_id, message, response)
        logger.info(f"Generated response for user {user_id}: {response[:100]}...") # Log response snippet
        return response
