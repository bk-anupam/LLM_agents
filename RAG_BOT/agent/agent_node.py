# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/agent_node.py
import json
import os
import sys
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from RAG_BOT.config import Config
from RAG_BOT.logger import logger
from RAG_BOT.agent.state import AgentState
from RAG_BOT.agent.prompts import FINAL_ANSWER_PROMPT


# Define the Pydantic Model for Structured Output ---
class FinalAnswerFormat(BaseModel):
    """Defines the structure for the final answer JSON."""
    answer: str = Field(description="The final answer to the user's query, based on the provided context and Brahmakumaris teachings.")


def agent_node(state: AgentState, llm_structured: ChatGoogleGenerativeAI, llm_with_tools: ChatGoogleGenerativeAI):
    """
    Handles initial query, decides first action, and generates final response.
    Now ensures final response adheres to JSON format defined in FINAL_ANSWER_PROMPT.
    """
    logger.info("--- Executing Agent Node ---")
    messages = state['messages']
    last_message = messages[-1]

    # 1. Handle Initial User Query
    if isinstance(last_message, HumanMessage):
        logger.info("Handling initial user query: " + last_message.content)
        original_query_content = last_message.content
        # Decide whether to retrieve context or answer directly (usually retrieve)
        system_prompt_msg = SystemMessage(content=Config.get_system_prompt())
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
            "context": None # Reset context
        }

    # 2. Generate Final Answer or "Cannot Find" Message (in JSON format)
    else:
        logger.info("Generating final response.")
        evaluation = state.get('evaluation_result')
        original_query = state.get('original_query')
        context = state.get('context') # Context should be populated by evaluate_context_node

        # Check if we should generate a proper answer
        # This happens if context is sufficient OR if agent decided to answer directly
        should_generate_answer = (evaluation == 'sufficient' or(evaluation is None and not state.get('retry_attempted')))
        if should_generate_answer:
            log_context_status = "sufficient" if evaluation == 'sufficient' else "answering directly (no evaluation)"
            logger.info(f"Context {log_context_status}. Generating final answer using structured output.")
            # Use base LLM without tools for response generation
            final_answer_chain = FINAL_ANSWER_PROMPT | llm_structured
            # Provide empty context if none was retrieved (direct answer case)
            structured_result  = final_answer_chain.invoke({
                "system_base": Config.get_system_prompt(), # Provide system_base here
                "original_query": original_query,
                "context": context if context else "N/A" # Provide N/A if no context
            })
            # Ensure the result is the expected Pydantic model instance
            if isinstance(structured_result, FinalAnswerFormat):
                logger.info(f"Successfully generated structured output: {structured_result}")
                # Convert the Pydantic model back to a JSON string for the AIMessage content
                # This maintains consistency if downstream code expects a JSON string.
                # Add ensure_ascii=False for broader character support if needed.
                final_content_json = structured_result.model_dump_json(indent=2)
                # Wrap in markdown code block for clarity in logs/output if desired
                final_content_for_message = f"```json\n{final_content_json}\n```"
                final_answer_message = AIMessage(content=final_content_for_message,
                                                 response_metadata=getattr(structured_result, 'response_metadata', {}))
                return {"messages": [final_answer_message]}
            else:
                # Handle unexpected output type (should be less likely with structured output)
                logger.error(f"Structured output LLM did not return the expected Pydantic model. Got: {type(structured_result)}")
                # Fallback to a generic error message, still formatted as JSON
                error_content = {"answer": "An internal error occurred while generating the final response."}
                error_json_string = f"```json\n{json.dumps(error_content, indent=2)}\n```"
                error_message = AIMessage(content=error_json_string)
                return {"messages": [error_message]}
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
