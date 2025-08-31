# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/agent_node.py
import json
from collections import defaultdict
from typing import Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.logger import logger
from RAG_BOT.src.agent.state import AgentState
from RAG_BOT.src.agent.prompts import get_final_answer_chat_prompt

def _generate_formatted_answer(state: AgentState, llm: ChatGoogleGenerativeAI, context_to_use: str) -> Dict[str, Any]:
    """
    A helper function to generate a final, formatted answer using a given context.
    This encapsulates the common logic for invoking the final answer chain.
    """
    language_code = state.get('language_code', 'en')
    mode = state.get('mode', 'default')
    original_query = state.get('original_query')

    final_answer_prompt = get_final_answer_chat_prompt(language_code, mode)
    final_answer_chain = final_answer_prompt | llm
    
    final_answer_llm_input = {
        "system_base": Config.get_system_prompt(language_code, mode),          
        "original_query": original_query,
        "context": context_to_use
    }
    logger.debug(f"Invoking final_answer_chain with input: {final_answer_llm_input}")
    final_answer = final_answer_chain.invoke(final_answer_llm_input)
    
    final_answer_content_str = getattr(final_answer, 'content', str(final_answer))
    if not isinstance(final_answer, AIMessage):
        final_answer = AIMessage(content=final_answer_content_str,
                                 response_metadata=getattr(final_answer, 'response_metadata', {}))
    return {"messages": [final_answer]}


def generate_final_response(state: AgentState, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """
    Generates and returns the final response based on retrieved context.
    Falls back to internal knowledge if context is insufficient.
    """
    logger.info("Generating final response ")
    evaluation = state.get('evaluation_result')    
    retry_attempted = state.get('retry_attempted', False)    
    # Determine the content to be used for the final answer.
    content_for_final_answer = state.get('retrieved_context')

    # If no context from retrieval, check if the last message was a direct AI answer.
    messages = state.get('messages', [])
    last_message_in_state = messages[-1] if messages else None
    if not content_for_final_answer and isinstance(last_message_in_state, AIMessage) and \
       not last_message_in_state.tool_calls and not getattr(last_message_in_state, 'tool_call_chunks', None):
        logger.info("No context from retrieval pipeline. Using content from the last direct AIMessage for final formatting.")
        content_for_final_answer = last_message_in_state.content
        logger.info(f"content_for_final_answer from last AIMessage:\n {str(content_for_final_answer)}")
    elif not content_for_final_answer and state.get('documents'): # If context is None but documents exist
        logger.info("Context is None, but documents found in state. Joining document contents for final answer.")
        content_for_final_answer = "\n\n".join([doc.page_content for doc in state['documents']])
    
    fallback_to_internal_knowledge = False
    if evaluation == 'insufficient' and retry_attempted:
        logger.info("Context evaluated as insufficient after retry. Falling back to internal knowledge.")
        fallback_to_internal_knowledge = True
    elif not content_for_final_answer and retry_attempted : # If retry was done, but still no content
        logger.info("Retry attempted, but no content available for final answer. Falling back to internal knowledge.")
        fallback_to_internal_knowledge = True
    # Add a case: if evaluation is None (no retrieval cycle for this query) AND no content_for_final_answer
    # This might happen if the first pass LLM decides not to call a tool and also doesn't provide an answer (unlikely with current prompts).
    elif evaluation is None and not content_for_final_answer and not retry_attempted:
         logger.info("Initial attempt, no evaluation, and no content for final answer. Falling back to internal knowledge.")
         fallback_to_internal_knowledge = True

    if not fallback_to_internal_knowledge and content_for_final_answer:
        logger.info(f"Proceeding to format final answer using content: '{str(content_for_final_answer)[:200]}...'")
        return _generate_formatted_answer(state, llm, content_for_final_answer)
    else:
        logger.info("Answering from internal knowledge about Brahma Kumaris philosophy.")
        # The key is to modify the context to be an instruction rather than retrieved data.
        no_context_instruction = (
            "No relevant context was found. Please answer the user's question based on your internal knowledge "
            "about Brahma Kumaris philosophy. If you are in research mode, provide an empty list for references."
        )
        return _generate_formatted_answer(state, llm, no_context_instruction)


async def handle_question_node(state: AgentState, llm_with_tools: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """
    Handles the initial user query, decides on the first action (tool call),
    and invokes the LLM with tools. This node is responsible for the initial
    interaction with the LLM to determine if a tool should be used.
    """
    logger.info(f"--- Executing Handle Question Node ---")
    messages = state.get('messages', [])
    last_message = messages[-1] if messages else None
    language_code = state.get('language_code', 'en')
    mode = state.get('mode', 'default')

    if not isinstance(last_message, HumanMessage):
        logger.error("Handle Question node was called without a HumanMessage as the last message. This is unexpected.")
        # Return a state that does nothing to avoid breaking the graph
        return {}

    logger.info(f"Handle Question node received HumanMessage: {last_message.content}")

    # Prepare messages for LLM with tools
    system_prompt_msg = SystemMessage(content=Config.get_system_prompt(language_code, mode))

    # We pass only the system prompt and the latest human message to ensure the LLM
    # strictly follows the tool-use rules without being influenced by past turns.
    # The full history is preserved in the state for the final answer generation step.
    tool_calling_prompt = [system_prompt_msg, last_message]

    logger.info("Invoking LLM for tool-calling decision with an isolated prompt.")
    response = await llm_with_tools.ainvoke(tool_calling_prompt)
    logger.info("LLM invoked for initial decision.")

    # Update state after LLM makes a decision (e.g., calls a tool or answers directly)
    updated_attrs = {
        "messages": [response],
        # For first invocation, set original_query. Afterwards, it's preserved.
        "original_query": state.get('original_query') or last_message.content,
        "current_query": last_message.content, # The query that was just processed
        "language_code": language_code,
        "mode": mode,
        # Reset fields for the new phase initiated by this HumanMessage/LLM response
        "evaluation_result": None,
        "documents": [], # Clear previous documents before new tool call
        "retrieved_context": None,   # Clear previous concatenated context
        "last_retrieval_source": None, # Will be set by next tool processing
        # Preserve retry_attempted if it was set by a preceding node (e.g., reframe_query_node)
        "retry_attempted": state.get('retry_attempted', False)
    }
    logger.info(f"Handle Question node updating state. Original query: '{updated_attrs['original_query']}', "
                f"Current query: '{updated_attrs['current_query']}'")
    return updated_attrs