# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/agent_node.py
import json
import os
import sys
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)
from RAG_BOT.config import Config
from RAG_BOT.logger import logger
from RAG_BOT.agent.state import AgentState
# Import the new prompt helper functions
from RAG_BOT.agent.prompts import get_final_answer_chat_prompt
from RAG_BOT import utils # Assuming utils has parse_json_answer


def agent_node(state: AgentState, llm: ChatGoogleGenerativeAI, llm_with_tools: ChatGoogleGenerativeAI):
    """
    Handles initial query, decides first action, and generates final response.
    Now ensures final response adheres to JSON format defined in FINAL_ANSWER_PROMPT.
    """
    logger.info("--- Executing Agent Node ---")
    messages = state['messages']
    last_message = messages[-1]
    language_code = state.get('language_code', 'en') # Default to English if not set

    # 1. Handle Initial User Query
    if isinstance(last_message, HumanMessage):
        logger.info("Handling initial user query: " + last_message.content)
        original_query_content = last_message.content
        # Decide whether to retrieve context or answer directly (usually retrieve)
        system_prompt_msg = SystemMessage(content=Config.get_system_prompt(language_code))
        # Use LLM with tools to decide if tool call is needed
        response = llm_with_tools.invoke([system_prompt_msg] + messages)
        logger.info("LLM invoked for initial decision.")
        # Update state for the first retrieval attempt
        return {
            "messages": [response],
            "original_query": original_query_content,
            "current_query": original_query_content, # Start with original query
            "retry_attempted": False,
            "evaluation_result": None, # Reset evaluation
            "context": None, # Reset context
            "language_code": language_code # Store language code
        }

    # 2. Generate Final Answer or "Cannot Find" Message (in JSON format)
    else:
        logger.info("Generating final response.")
        evaluation = state.get('evaluation_result')
        original_query = state.get('original_query')
        context = state.get('context')         

        # Check if we should generate a proper answer
        # This happens if context is sufficient OR if agent decided to answer directly
        should_generate_answer = (evaluation == 'sufficient' or(evaluation is None and not state.get('retry_attempted')))
        if should_generate_answer:
            log_context_status = "sufficient" if evaluation == 'sufficient' else "answering directly (no evaluation)"
            logger.info(f"Context {log_context_status}. Generating final answer.")
            # Use base LLM without tools for response generation
            final_answer_prompt = get_final_answer_chat_prompt(language_code) 
            logger.debug(f"Final answer prompt:  {final_answer_prompt}")
            final_answer_chain = final_answer_prompt | llm
            # Provide empty context if none was retrieved (direct answer case)
            final_answer  = final_answer_chain.invoke({      
                "system_base": Config.get_system_prompt(language_code),          
                "original_query": original_query,
                "context": context if context else "N/A" # Provide N/A if no context
            })
            if not isinstance(final_answer, AIMessage):
                final_answer = AIMessage(content=str(final_answer.content),
                                        response_metadata=getattr(final_answer, 'response_metadata', {}))
            return {"messages": [final_answer]}
        else: # Evaluation was insufficient after retry, or some other error state
            logger.info("Context insufficient after retry or error. Generating 'cannot find' message in JSON format.")
            # Format the "cannot find" message as JSON
            cannot_find_content = {
                "answer": "Relevant information cannot be found in the database to answer the question. Please try reframing your question."
            }
            # Wrap in markdown code block as per the prompt's example output
            cannot_find_json_string = f"```json\n{json.dumps(cannot_find_content, indent=2)}\n```"
            cannot_find_message = AIMessage(content=cannot_find_json_string)
            return {"messages": [cannot_find_message]}
