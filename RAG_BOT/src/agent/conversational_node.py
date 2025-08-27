from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from RAG_BOT.src.logger import logger
from RAG_BOT.src.agent.state import AgentState
from RAG_BOT.src.config.config import Config


async def conversational_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """
    Handles conversational queries using full conversation history.    
    This node has access to the complete message history and can answer
    questions about the conversation itself, provide summaries, handle
    greetings, and engage in general chat.
    
    Args:
        state: Current agent state with full message history
        llm: Language model for generating responses
        
    Returns:
        Dict containing the conversational response
    """
    logger.info("--- Executing Conversational Node ---")
    
    messages = state.get('messages', [])
    language_code = state.get('language_code', 'en')
    mode = state.get('mode', 'default')
    
    if not messages:
        logger.warning("No messages found for conversational response")
        return {"messages": [AIMessage(content="```json\\n{\\\"answer\\\": \\\"I don't have any conversation history to work with.\\\"}\\n```")]}
    
    # Get the last user message
    last_message = messages[-1] if messages else None
    user_query = last_message.content if isinstance(last_message, HumanMessage) else ""    
    logger.info(f"Processing conversational query: '{user_query[:100]}...' (Language: {language_code}, Mode: {mode})")
    
    try:
        # Create conversational prompt with full history
        system_prompt = Config.get_conversational_system_prompt(language_code)
        conversational_messages = [SystemMessage(content=system_prompt)] + messages        
        # Generate response using full conversation context
        response = await llm.ainvoke(conversational_messages)        
        # Ensure we have an AIMessage response
        if not isinstance(response, AIMessage):
            response_content = getattr(response, 'content', str(response))
            response = AIMessage(
                content=response_content,
                response_metadata=getattr(response, 'response_metadata', {})
            )
        
        logger.info(f"Generated conversational response: {response.content[:100]}...")        
        return {"messages": [response]}    
        
    except Exception as e:
        logger.error(f"Error in conversational_node: {e}", exc_info=True)        
        # Generate error response in appropriate language
        error_message = "Sorry, I encountered an error while processing your conversational request."
        if language_code == 'hi':
            error_message = "क्षमा करें, आपके वार्तालाप संबंधी अनुरोध को संसाधित करते समय मुझे एक त्रुटि का सामना करना पड़ा।"
        
        error_response = f'```json\\n{{"answer": "{error_message}"}}\\n```'
        return {"messages": [AIMessage(content=error_response)]}