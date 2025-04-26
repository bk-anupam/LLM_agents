# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/retrieval_nodes.py
import os
import sys
from operator import itemgetter
from typing import List

from langchain_core.messages import ToolMessage
from sentence_transformers import CrossEncoder

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from RAG_BOT.config import Config
from RAG_BOT.logger import logger
from RAG_BOT.agent.state import AgentState


def rerank_context_node(state: AgentState, reranker_model: CrossEncoder):
    """
    Reranks the initially retrieved documents based on the current query.
    Updates the state with the final, concatenated context string.
    """
    logger.info("--- Executing Rerank Context Node ---")
    current_query = state.get('current_query')
    logger.info(f"Current query for reranking: '{current_query}'")
    messages = state.get('messages')
    last_message = messages[-1] if messages else None
    logger.info(f"Last message type: {type(last_message)}")
    # retrieved_docs = last_message.content
    # logger.info(f"Retrieved docs type: {type(retrieved_docs)}")
    # logger.info(f"isinstance(retrieved_docs, list): {isinstance(retrieved_docs, list)}")
    # logger.info(f"Last message content: {last_message.content if last_message else 'None'}")

    retrieved_docs_artifact = None
    # *** KEY CHANGE: Access the artifact from the last message ***
    if isinstance(last_message, ToolMessage) and hasattr(last_message, 'artifact'):
        # Optional: Check if it's the correct tool message
        if last_message.name == 'retrieve_context':
            retrieved_docs_artifact = last_message.artifact
            logger.info(f"Artifact received from tool '{last_message.name}': type {type(retrieved_docs_artifact)}")
        else:
            logger.error(f"Last message is a ToolMessage, but not from 'retrieve_context' (name: {last_message.name}). Skipping reranking.")
            # Skip reranking if not from the correct tool
            return {"context": None}
    else:
        logger.warning(f"Last message is not a ToolMessage with an artifact (type: {type(last_message)}). Skipping reranking.")
        return {"context": None}

    # Proceed only if we got a valid list artifact
    if not isinstance(retrieved_docs_artifact, list):
        logger.error("No valid document list artifact found to rerank.")
        # Set context to empty or None depending on desired downstream handling
        return {"context": ""} # Or {"context": None}

    # Validate reranker and input
    if reranker_model is None:
        logger.error("Reranker model not loaded. Concatenating original artifact docs.")
        final_context = "\n\n".join(retrieved_docs_artifact)
        return {"context": final_context}

    if not retrieved_docs_artifact:
        logger.info("No documents in artifact to rerank.")
        return {"context": ""}

    logger.info(f"Reranking {len(retrieved_docs_artifact)} documents for query: '{current_query}'")
    # Prepare pairs for the cross-encoder
    pairs = [[current_query, doc] for doc in retrieved_docs_artifact]
    try:
        # Get scores from the cross-encoder
        scores = reranker_model.predict(pairs)
        logger.info(f"Reranking scores obtained (Top {Config.RERANK_TOP_N}): {scores[:Config.RERANK_TOP_N]}")
        # Combine docs with scores and sort
        scored_docs = list(zip(scores, retrieved_docs_artifact))
        scored_docs.sort(key=itemgetter(0), reverse=True)
        # Select top N documents based on config
        reranked_docs = [doc for score, doc in scored_docs[:Config.RERANK_TOP_N]]
        logger.info(f"Selected top {len(reranked_docs)} documents after reranking.")
        # Concatenate the final context
        final_context = "\n\n".join(reranked_docs)
        logger.info(f"Final reranked context snippet: {final_context[:400]}...")
        # Update state with the final context
        # Optionally store raw docs too: "raw_retrieved_docs": retrieved_docs
        return {"context": final_context}
    except Exception as e:
        logger.error(f"Error during reranking: {e}. Using original context without reranking", exc_info=True)
        # Fallback: use the original concatenated context
        final_context = "\n\n".join(retrieved_docs_artifact)
        return {"context": final_context}
