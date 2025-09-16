from typing import Literal
from langchain_core.messages import AIMessage
from RAG_BOT.src.logger import logger
from RAG_BOT.src.agent.state import AgentState

# --- Conditional Edge Logic ---

def decide_next_step_after_evaluation(state: AgentState) -> Literal["reframe_query", "agent_final_answer"]:
    """
    Determines the next node based on evaluation result and retry status.    
    """
    logger.info("--- Deciding Next Step After Evaluation ---")
    evaluation = state.get('evaluation_result')
    retry_attempted = state.get('retry_attempted', False)
    logger.info(f"Evaluation: {evaluation}, Retry Attempted: {retry_attempted}")
    if evaluation == "sufficient":
        logger.info("Decision: Context sufficient, proceed to final answer generation.")
        return "agent_final_answer"
    elif not retry_attempted:
        logger.info("Decision: Context insufficient, attempt to reframe query.")
        return "reframe_query"
    else:
        logger.info("Decision: Context insufficient after retry, proceed to 'cannot find' message.")
        return "agent_final_answer"
    

def should_invoke_tool_after_web_search_force(state: AgentState) -> Literal["tool_invoker", "agent_final_answer"]:
    """
    Checks if force_web_search_node produced a tool call.
    If yes, invoke the tool.
    If no (e.g., no date found), go to the final answer node to prevent loops.
    """
    logger.info("--- Deciding whether to invoke tool after force_web_search ---")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Check if the tool call was added by force_web_search
        if any(tc.get("id") == "tool_call_forced_web_search" for tc in last_message.tool_calls):
            logger.info("Decision: Tool call found from force_web_search. Invoking tool.")
            return "tool_invoker"

    logger.info("Decision: No tool call from force_web_search. Proceeding to final answer to prevent loop.")
    return "agent_final_answer"


def route_after_retrieval(state: AgentState) -> Literal["rerank_context", "force_web_search", "agent_final_answer"]:
    """
    Decides the next step after tool output has been processed.
    - If documents found: rerank.
    - If local search failed and web not tried: go to force_web_search_node.
    - Else (web search failed or no other options): go to final answer (likely "cannot find").
    """
    logger.info("--- Routing After Retrieval Attempt ---")
    documents = state.get("documents")
    last_retrieval_source = state.get("last_retrieval_source")
    web_search_attempted = state.get("web_search_attempted", False)

    logger.info(f"Routing with: docs_count={len(documents) if documents else 0}, last_retrieval_source='{last_retrieval_source}', "
                f"web_search_attempted={web_search_attempted}")

    if documents and len(documents) > 0:
        logger.info(f"Documents found ({len(documents)}) from '{last_retrieval_source}'. Proceeding to rerank.")
        return "rerank_context"
    else: # No documents found
        logger.info(f"No documents found from '{last_retrieval_source}' retrieval attempt.")
        if last_retrieval_source == "local" and not web_search_attempted:
            logger.info("Local retrieval failed, web search not yet attempted. Routing to force_web_search.")
            return "force_web_search"
        else:
            logger.info("Web search already attempted and failed, or was the primary failed source, or agent decided against "
                        f"web search. Proceeding to final answer.")
            return "agent_final_answer"