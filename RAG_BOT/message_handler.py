import datetime
from config import Config
from logger import logger
from langchain_core.messages import HumanMessage
from telebot.types import Message
from langgraph.graph import StateGraph


class MessageHandler:
    def __init__(self, agent: StateGraph, config: Config):
        # Store user session data (could be moved to a database for persistence)
        self.config = config
        self.sessions = config.USER_SESSIONS
        self.agent = agent

    def _get_user_session(self, user_id):
        """Get or create a new session for the user"""
        if user_id not in self.sessions:
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
        if len(session['conversation']) > 10:
            session['conversation'] = session['conversation'][-10:]

    def process_message(self, incoming_message: Message):
        """
        Process the incoming message and generate a response
        This is where you implement your custom logic
        """
        user_id = incoming_message.from_user.id
        message = incoming_message.text        
        logger.info(f"Received message from {user_id}: {message}")        
        # Get user session
        session = self._get_user_session(user_id)        
        # Convert message to lowercase for easier matching
        message_lower = message.lower().strip()        
        # Basic response logic
        if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey']):
            response = "👋 Hello! I'm your Telegram assistant. How can I help you today?"        
        elif "help" in message_lower:
            response = ("Here's what I can do:\n"
                    "- Answer your query (on pdf documents uploaded). Use /query command followed by the query for this\n"
                    "- index and store in vector DB uploaded pdf documents. Just send the pdf document as a message\n"
                    "- answer any general query \n" 
                    "- last message - to see your last message\n"                     
                    "Just let me know what you need!")                        
        elif "last message" in message_lower:
            if len(session['conversation']) > 0:
                last_message = session['conversation'][-1]['user']
                response = f"Your last message was: '{last_message}'"
            else:
                response = "You haven't sent any previous messages."        
        else:            
            try:
                logger.info(f"thread_id = {str(incoming_message.chat.id)}, query = {message}")
                config = {"configurable": {"thread_id": str(incoming_message.chat.id)}}
                # Build the initial state for the agent
                initial_state = {"messages": [HumanMessage(content=message)]}
                final_state = self.agent.invoke(initial_state, config)
                answer = None
                if isinstance(final_state, dict) and "messages" in final_state and final_state["messages"]:
                    last_msg = final_state["messages"][-1]
                    if hasattr(last_msg, "content"):
                        answer = last_msg.content
                if not answer:
                    answer = "Sorry, I couldn't find an answer."
                response = answer
            except Exception as e:
                logger.error(f"Error invoking RAG agent: {str(e)}")
                response = "Sorry, I encountered an error processing your query." 
        # Update session with this interaction
        self._update_session(user_id, message, response)        
        return response