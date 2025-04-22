# /home/bk_anupam/code/LLM_agents/RAG_BOT/message_handler.py
import datetime
from config import Config
from RAG_BOT.logger import logger # Corrected import path based on other files
from langchain_core.messages import HumanMessage
from telebot.types import Message
from langgraph.graph import StateGraph


class MessageHandler:
    def __init__(self, agent: StateGraph, config: Config, vector_store=None): # Added vector_store based on bot.py usage
        # Store user session data (could be moved to a database for persistence)
        self.config = config
        self.sessions = config.USER_SESSIONS
        self.agent = agent
        self.vector_store = vector_store # Store vector_store if needed for direct queries

    def _get_user_session(self, user_id):
        """Get or create a new session for the user"""
        if user_id not in self.sessions:
            logger.info(f"Creating new session for user {user_id}") # Added log
            self.sessions[user_id] = {
                'last_interaction': datetime.datetime.now(),
                'conversation': [],
                'context': {}
            }
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


    def process_message(self, incoming_message: Message):
        """
        Process the incoming message and generate a response
        This is where you implement your custom logic
        """
        user_id = incoming_message.from_user.id
        message = incoming_message.text
        if not message: # Handle cases like stickers or empty messages if needed
             logger.warning(f"Received empty message text from user {user_id}")
             return "Sorry, I didn't receive any text."

        logger.info(f"Processing message from {user_id}: {message[:100]}...") # Log snippet
        # Get user session
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
                logger.info(f"Invoking agent for thread_id={str(incoming_message.chat.id)}, query='{message[:50]}...'")
                config_thread = {"configurable": {"thread_id": str(incoming_message.chat.id)}}
                # Build the initial state for the agent
                initial_state = {"messages": [HumanMessage(content=message)]}

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
                            answer = last_msg.content
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

