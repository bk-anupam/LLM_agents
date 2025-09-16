from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from RAG_BOT.src.logger import logger
from RAG_BOT.src.agent.state import AgentState
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.agent.prompts import get_conversational_chat_prompt
from RAG_BOT.src.utils import get_conversational_history


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
        # Filter the history to remove tool calls and tool messages for a cleaner context
        clean_history = get_conversational_history(messages)
        logger.info(f"Using cleaned history with {len(clean_history)} messages for conversational response.")

        # Pop the last human message to append a more forceful language instruction.
        # This is crucial when the conversation history (e.g., a summary) is in a different
        # language, as it helps override the LLM's tendency to follow the history's language.
        if clean_history:
            last_human_message = clean_history.pop()
            lang_instruction = Config.get_final_answer_language_instruction(language_code)

            if isinstance(last_human_message, HumanMessage) and lang_instruction:
                # This places the instruction at the very end of the prompt, which is highly influential.
                modified_content = f"{last_human_message.content}\n\n---\n{lang_instruction}"
                modified_human_message = HumanMessage(content=modified_content)
                clean_history.append(modified_human_message)
                logger.info("Appended language instruction to the final user query for conversational node.")
            else:
                # If it's not a HumanMessage or no lang_instruction, add it back unmodified.
                clean_history.append(last_human_message)

        # Get the prompt template
        conversational_prompt = get_conversational_chat_prompt()        
        # Prepare values for the prompt template
        prompt_values = {
            "lang_instruction": Config.get_final_answer_language_instruction(language_code),
            "json_format_instructions": Config.get_json_format_instructions(),
            "history": clean_history
        }
        # Create the full chain
        conversational_chain = conversational_prompt | llm
        # Log the messages that will be sent to the LLM for debugging
        final_messages_for_llm = conversational_prompt.invoke(prompt_values).to_messages()
        for i, msg in enumerate(final_messages_for_llm):
            role = "Human" if isinstance(msg, HumanMessage) else "AI" if isinstance(msg, AIMessage) else "System"
            logger.debug(f"Conversational Message {i} ({role}): {msg.content}")
        
        # Generate response using the chain
        response = await conversational_chain.ainvoke(prompt_values)
        
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
        error_message = "Sorry, I encountered an error while processing your conversational request."
        if language_code == 'hi':
            error_message = "क्षमा करें, आपके वार्तालाप संबंधी अनुरोध को संसाधित करते समय मुझे एक त्रुटि का सामना करना पड़ा।"
        
        error_response = f'```json\\n{{"answer": "{error_message}"}}\\n```'
        return {"messages": [AIMessage(content=error_response)]}