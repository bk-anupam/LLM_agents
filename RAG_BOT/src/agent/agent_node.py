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


def generate_final_response(state: AgentState, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """
    Generates and returns the final response based on the evaluation result,
    original query, and retrieved context.
    """
    logger.info("Generating final response ")
    evaluation = state.get('evaluation_result')
    original_query = state.get('original_query')
    retry_attempted = state.get('retry_attempted', False)
    language_code = state.get('language_code', 'en')
    mode = state.get('mode', 'default') # Get the mode from the state

    # Determine the content to be used for the final answer.
    content_for_final_answer = state.get('retrieved_context')
    # reranked documents used as context for final answer
    context_docs = state.get('documents', [])

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
    
    generate_cannot_find = False
    if evaluation == 'insufficient' and retry_attempted:
        logger.info("Context evaluated as insufficient after retry. Generating 'cannot find' message.")
        generate_cannot_find = True
    elif not content_for_final_answer and retry_attempted : # If retry was done, but still no content
        logger.info("Retry attempted, but no content available for final answer. Generating 'cannot find' message.")
        generate_cannot_find = True
    # Add a case: if evaluation is None (no retrieval cycle for this query) AND no content_for_final_answer
    # This might happen if the first pass LLM decides not to call a tool and also doesn't provide an answer (unlikely with current prompts).
    elif evaluation is None and not content_for_final_answer and not retry_attempted:
         logger.info("Initial attempt, no evaluation, and no content for final answer. Generating 'cannot find' message.")
         generate_cannot_find = True

    if not generate_cannot_find and content_for_final_answer:
        logger.info(f"Proceeding to format final answer using content: '{str(content_for_final_answer)[:200]}...'")
        # Pass the mode to get the correct prompt template
        final_answer_prompt = get_final_answer_chat_prompt(language_code, mode)
        final_answer_chain = final_answer_prompt | llm
        
        final_answer_llm_input = {
            "system_base": Config.get_system_prompt(language_code, mode),          
            "original_query": original_query,
            "context": content_for_final_answer
        }
        logger.debug(f"Invoking final_answer_chain with input: {final_answer_llm_input}")
        final_answer = final_answer_chain.invoke(final_answer_llm_input)
        
        final_answer_content_str = getattr(final_answer, 'content', str(final_answer))
        if not isinstance(final_answer, AIMessage):
            final_answer = AIMessage(content=final_answer_content_str,
                                     response_metadata=getattr(final_answer, 'response_metadata', {}))
        return {"messages": [final_answer]}
    else:
        logger.info("Context insufficient after retry or error. Generating 'cannot find' message in JSON format.")
        # The "cannot find" message should also respect the mode if it implies a different format
        # For simplicity, we'll keep it consistent for now, but could be made mode-aware.
        cannot_find_content = {
            "answer": "Relevant information cannot be found in the database to answer the question. Please try reframing your question."
        }
        # If in research mode, add an empty references array
        if mode == 'research':
            cannot_find_content["references"] = []
            
        cannot_find_json_string = f"```json\n{json.dumps(cannot_find_content, indent=2, ensure_ascii=False)}\n```"
        cannot_find_message = AIMessage(content=cannot_find_json_string)
        return {"messages": [cannot_find_message]}


async def agent_node(state: AgentState, llm: ChatGoogleGenerativeAI, llm_with_tools: ChatGoogleGenerativeAI):
    """
    Handles initial query, decides first action, and generates final response.
    Now ensures final response adheres to JSON format defined in FINAL_ANSWER_PROMPT.
    """    
    logger.info(f"--- Executing Agent Node ---") 
    messages = state.get('messages', []) 
    last_message = messages[-1] if messages else None 
    language_code = state.get('language_code', 'en')
    mode = state.get('mode', 'default') 

    # 1. Handle Initial User Query
    if isinstance(last_message, HumanMessage):
        logger.info(f"Agent node received HumanMessage: {last_message.content}")        
        # Check if this is a retry scenario after local retrieval failed
        # This logic assumes agent_initial is re-entered after local failure
        # and before web search is attempted.        
        current_query_for_guidance = state.get("current_query") or last_message.content # Ensure we have a query for guidance
        documents_from_state = state.get("documents")
        last_retrieval_source = state.get("last_retrieval_source")
        web_search_attempted = state.get("web_search_attempted", False)
        prompt_prefix_messages = []
        if last_retrieval_source == "local" and \
           (not documents_from_state or len(documents_from_state) == 0) and not web_search_attempted:
                        
            logger.info(f"Agent node entered after local retrieval failed for query '{current_query_for_guidance}'." 
                         f"Web search attempted: {web_search_attempted}. Relying on graph logic for web search if needed.")

        # Prepare messages for LLM with tools
        system_prompt_msg = SystemMessage(content=Config.get_system_prompt(language_code, mode))
                
        # We pass only the system prompt and the latest human message to ensure the LLM
        # strictly follows the tool-use rules without being influenced by past turns.
        # The full history is preserved in the state for the final answer generation step.
        tool_calling_prompt = [system_prompt_msg] + prompt_prefix_messages + [last_message]
        
        logger.info("Invoking LLM for tool-calling decision with an isolated prompt.")
        response = await llm_with_tools.ainvoke(tool_calling_prompt)
        logger.info("LLM invoked for initial decision.")

        # Update state after LLM makes a decision (e.g., calls a tool or answers directly)
        updated_attrs = {
            "messages": [response],
            # For first invocation of agent_node, we set original_query to last_message.content and 
            # afterwards extract it from agent state.
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
        logger.info(f"Agent node updating state. Original query: '{updated_attrs['original_query']}', "
                    f"Current query: '{updated_attrs['current_query']}'")
        return updated_attrs
    # 2. Generate Final Answer (handles direct AI responses or context-based answers)    
    else:
        return generate_final_response(state, llm)