# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/evaluation_nodes.py
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage 
from RAG_BOT.logger import logger
from RAG_BOT.agent.state import AgentState
from RAG_BOT.agent.prompts import get_evaluate_context_chat_prompt, get_reframe_question_chat_prompt


def evaluate_context_node(state: AgentState, llm: ChatGoogleGenerativeAI):
    """
    Evaluates the reranked context (read from state) based on the original query.
    """
    logger.info("--- Executing Evaluate Context Node ---")
    original_query = state.get('original_query')
    # Context is now expected to be populated by the rerank_context_node
    context_to_evaluate = state.get('context') 

    # Check if context exists in the state
    if context_to_evaluate is None: 
         logger.warning("Evaluate context node reached but no context found in state "
                        "(potentially due to retrieval/reranking failure). Treating as insufficient.")
         return {"evaluation_result": "insufficient"} 

    if not original_query:
        logger.warning("Missing original query for evaluation.")        
        return {"evaluation_result": "insufficient"}

    # Handle empty string context (e.g., if reranker returned no docs)
    if not context_to_evaluate:
        logger.info("Context in state is empty. Treating as insufficient.")
        return {"evaluation_result": "insufficient"}

    # Check for error messages passed as context (though reranker should return None on error)
    if isinstance(context_to_evaluate, str) and context_to_evaluate.startswith("Error:"):
        logger.warning(f"Context contains an error message: {context_to_evaluate}. Treating as insufficient.")
        return {"evaluation_result": "insufficient"}

    logger.info("Evaluating context from state...")
    logger.debug(f"Context to evaluate: {context_to_evaluate}")
    eval_chain = get_evaluate_context_chat_prompt() | llm
    evaluation_result_str = eval_chain.invoke({
        "original_query": original_query,
        "context": context_to_evaluate # Use context from state
    }).content.strip().upper()

    logger.info(f"Context evaluation result: {evaluation_result_str}")
    evaluation = "sufficient" if evaluation_result_str == "YES" else "insufficient"

    # Context is already in state, just return the evaluation result
    return {"evaluation_result": evaluation}


def reframe_query_node(state: AgentState, llm: ChatGoogleGenerativeAI):
    """
    Reframes the query if the first retrieval attempt was insufficient.
    """
    logger.info("--- Executing Reframe Query Node ---")
    original_query = state['original_query']
    # The query used in the failed attempt
    failed_query = state['current_query']

    if not original_query or not failed_query:
        logger.error("Missing original or current query for reframing.")
        # Handle error state - perhaps skip reframing and go to final "cannot find"
        # For now, just update state to prevent infinite loops
        return {"retry_attempted": True, "current_query": original_query} # Fallback

    logger.info("Reframing question...")
    reframe_chain = get_reframe_question_chat_prompt() | llm
    reframed_question = reframe_chain.invoke({
        "original_query": original_query, # The very first user query
        "failed_query": failed_query # The query that just failed to yield good results
    }).content.strip()

    logger.info(f"Reframed question: {reframed_question}")

    # The reframe_query_node now just updates the query and state for retry.
    # It does NOT create a tool call message.
    # The graph will route back to agent_initial, which will decide on the tool.
    # We need to add a HumanMessage to the messages list that represents this new query turn.
    # This is important for the agent_node to process it as a new input.
    return {
        # Pass the reframed query as a new HumanMessage
        "messages": state['messages'] + [HumanMessage(content=reframed_question)], 
        "current_query": reframed_question,
        "retry_attempted": True,
        # Reset evaluation for the next attempt
        "evaluation_result": None, 
        # Clear documents before retry
        "documents": [], 
        # Clear concatenated context string
        "context": None, 
        # Reset flags for the new reframed query attempt
        "web_search_attempted": False,
        "last_retrieval_source": None,
    }
