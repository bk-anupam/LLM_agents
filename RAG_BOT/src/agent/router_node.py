# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/router_node.py
import functools
from typing import Dict, Any, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from RAG_BOT.src.logger import logger
from RAG_BOT.src.agent.state import AgentState
from RAG_BOT.src.config.config import Config


async def router_node(state: AgentState, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """
    Routes user queries to either RAG path or conversational path.
    
    Args:
        state: Current agent state
        llm: Language model for classification
        
    Returns:
        Dict containing the routing decision
    """
    logger.info("--- Executing Router Node ---")
    
    messages = state.get('messages', [])
    if not messages:
        logger.warning("No messages found in state for routing")
        return {"route_decision": "RAG_QUERY"}
    
    # Get the last human message
    last_message = messages[-1] if messages else None
    if not isinstance(last_message, HumanMessage):
        logger.warning("Last message is not a HumanMessage, defaulting to RAG_QUERY")
        return {"route_decision": "RAG_QUERY"}
    
    user_query = last_message.content
    logger.info(f"Routing query: '{user_query[:100]}...'")
    
    try:
        # Create classification prompt
        classification_messages = [
            SystemMessage(content=Config.get_router_system_prompt()),
            HumanMessage(content=user_query)
        ]
        
        # Get classification from LLM
        response = await llm.ainvoke(classification_messages)
        classification = response.content.strip().upper()
        
        # Validate classification
        if classification not in ["RAG_QUERY", "CONVERSATIONAL_QUERY"]:
            logger.warning(f"Invalid classification '{classification}', defaulting to RAG_QUERY")
            classification = "RAG_QUERY"
        
        logger.info(f"Query classified as: {classification}")
        return {"route_decision": classification}
        
    except Exception as e:
        logger.error(f"Error in router_node: {e}", exc_info=True)
        # Default to RAG_QUERY on error
        return {"route_decision": "RAG_QUERY"}


def route_query_decision(state: AgentState) -> Literal["rag_path", "conversational_path"]:
    """
    Conditional edge function that determines the path based on router decision.
    
    Args:
        state: Current agent state
        
    Returns:
        Path name for the next step
    """
    route_decision = state.get("route_decision", "RAG_QUERY")
    
    if route_decision == "RAG_QUERY":
        logger.info("Routing to RAG path for knowledge-based query")
        return "rag_path"
    else:
        logger.info("Routing to conversational path for chat-based query")
        return "conversational_path"