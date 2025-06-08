# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/agent_node.py
import json
from typing import Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)
from RAG_BOT.config import Config
from RAG_BOT.logger import logger
from RAG_BOT.agent.state import AgentState
from RAG_BOT.agent.prompts import get_final_answer_chat_prompt


def generate_final_response(state: AgentState, llm: ChatGoogleGenerativeAI, language_code: str) -> Dict[str, Any]:
    """
    Generates and returns the final response based on the evaluation result,
    original query, and retrieved context.
    """
    logger.info("Generating final response.")
    evaluation = state.get('evaluation_result')
    original_query = state.get('original_query')
    # Get reranked context or join documents if not set
    final_context_str = state.get('context')
    if final_context_str is None and state.get('documents'):
        final_context_str = "\n\n".join([doc.page_content for doc in state['documents']])
    
    # Decide if a proper answer should be generated.
    # This happens if context is sufficient OR if agent decided to answer directly
    should_generate_answer = (evaluation == 'sufficient' or (evaluation is None and not state.get('retry_attempted')))
    if should_generate_answer:
        log_context_status = "sufficient" if evaluation == 'sufficient' else "answering directly (no evaluation)"
        logger.info(f"Context {log_context_status}. Generating final answer.")
        final_answer_prompt = get_final_answer_chat_prompt(language_code)
        logger.debug(f"Final answer prompt:  {final_answer_prompt}")
        final_answer_chain = final_answer_prompt | llm
        final_answer = final_answer_chain.invoke({      
            "system_base": Config.get_system_prompt(language_code),          
            "original_query": original_query,
            "context": final_context_str
        })
        if not isinstance(final_answer, AIMessage):
            final_answer = AIMessage(content=str(final_answer.content),
                                     response_metadata=getattr(final_answer, 'response_metadata', {}))
        return {"messages": [final_answer]}
    else:
        logger.info("Context insufficient after retry or error. Generating 'cannot find' message in JSON format.")
        cannot_find_content = {
            "answer": "Relevant information cannot be found in the database to answer the question. Please try reframing your question."
        }
        cannot_find_json_string = f"```json\n{json.dumps(cannot_find_content, indent=2)}\n```"
        cannot_find_message = AIMessage(content=cannot_find_json_string)
        return {"messages": [cannot_find_message]}


async def agent_node(state: AgentState, llm: ChatGoogleGenerativeAI, llm_with_tools: ChatGoogleGenerativeAI):
    """
    Handles initial query, decides first action, and generates final response.
    Now ensures final response adheres to JSON format defined in FINAL_ANSWER_PROMPT.
    """
    # Helps trace which agent node call
    logger.info(f"--- Executing Agent Node ---") 
    messages = state['messages']
    last_message = messages[-1]
    language_code = state.get('language_code', 'en')
    current_query = state.get("current_query", "") 
    documents_from_state = state.get("documents")
    last_retrieval_source = state.get("last_retrieval_source")
    web_search_attempted = state.get("web_search_attempted", False) 

    # 1. Handle Initial User Query
    if isinstance(last_message, HumanMessage):
        logger.info(f"Agent node received HumanMessage: {last_message.content}")        
        # Check if this is a retry scenario after local retrieval failed
        # This logic assumes agent_initial is re-entered after local failure
        # and before web search is attempted.
        # THIS GUIDANCE LOGIC IS NOW LESS CRITICAL IF force_web_search_node is used,
        # but can be kept as a fallback or if agent_initial is entered under other similar conditions.
        prompt_prefix_messages = []
        if last_retrieval_source == "local" and \
           (not documents_from_state or len(documents_from_state) == 0) and not web_search_attempted:
            
            # The explicit guidance prompt might still be useful if, for some reason,
            # the flow reaches agent_initial instead of force_web_search_node.
            # formatted_date_for_url = utils.extract_date_from_text(current_query, return_date_format="%d.%m.%y")            
            # if formatted_date_for_url:                
            #     guidance_content = Config.get_guidance_prompt(language_code, current_query, formatted_date_for_url)
            #     logger.info(f"guidance_content (in agent_node): {guidance_content}")
            #     prompt_prefix_messages.append(SystemMessage(content=guidance_content))
            logger.info(f"Agent node entered after local retrieval failed. Web search attempted: {web_search_attempted}. Relying on graph logic for web search if needed.")

        # Prepare messages for LLM with tools
        system_prompt_msg = SystemMessage(content=Config.get_system_prompt(language_code))
        # Use LLM with tools to decide if tool call is needed
        # The 'messages' in state already includes the last_message (HumanMessage)
        # prompt_prefix_messages are inserted before the main history.
        response = await llm_with_tools.ainvoke([system_prompt_msg] + prompt_prefix_messages + messages)
        logger.info("LLM invoked for initial decision.")

        # Update state after LLM makes a decision (e.g., calls a tool or answers directly)
        updated_attrs = {
            "messages": [response],
            # For first invocation of agent_node, we set original_query to last_message.content and 
            # afterwards extract it from agent state.
            "original_query": state.get('original_query') or last_message.content,
            "current_query": last_message.content, # The query that was just processed
            "language_code": language_code,
            # Reset fields for the new phase initiated by this HumanMessage/LLM response
            "evaluation_result": None,
            "documents": [], # Clear previous documents before new tool call
            "context": None,   # Clear previous concatenated context
            "last_retrieval_source": None, # Will be set by next tool processing
            # web_search_attempted is now set by force_web_search_node or process_tool_output
            # "web_search_attempted": False, 
            # Preserve retry_attempted if it was set by a preceding node (e.g., reframe_query_node)
            "retry_attempted": state.get('retry_attempted', False)
        }
        logger.info(f"Agent node updating state. Original query: '{updated_attrs['original_query']}', Current query: '{updated_attrs['current_query']}'")
        return updated_attrs
    # 2. Generate Final Answer or "Cannot Find" Message (in JSON format)
    else:
        return generate_final_response(state, llm, language_code)
