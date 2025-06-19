# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/retrieval_nodes.py
import os
import sys
from operator import itemgetter
from typing import List, Optional 
from langchain_core.messages import ToolMessage
from sentence_transformers import CrossEncoder
from RAG_BOT.config import Config
from RAG_BOT.logger import logger
from RAG_BOT.agent.state import AgentState


def rerank_context_node(state: AgentState, reranker_model: CrossEncoder, app_config: Config):
    """
    Reranks the initially retrieved documents based on the current query.
    Updates the state with the final, concatenated context string.
    """    
    logger.info("--- Executing Rerank Context Node ---")
    current_query = state.get('current_query')     
    retrieved_documents: Optional[List] = state.get('documents')
    logger.info(f"Current query for reranking: '{current_query}'")
    messages = state.get('messages')
    last_message = messages[-1] if messages else None
    logger.info(f"Last message type: {type(last_message)}")       

    if not retrieved_documents:
        logger.info("No documents found in state to rerank.")
        return {"context": "", "documents": []} 

    # Extract page content for reranking
    doc_contents_for_reranking = []
    for doc in retrieved_documents:
        if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
            doc_contents_for_reranking.append(doc.page_content)
        else:
            logger.warning(f"Document object without string page_content found: {doc}")

    if not doc_contents_for_reranking:
        logger.info("No document contents extracted for reranking.")
        return {"context": "", "documents": retrieved_documents} # Keep original docs if no content

    # Validate reranker and input
    if reranker_model is None:
        logger.warning("Reranker model not loaded. Concatenating original document contents.")
        final_context = "\n\n".join(doc_contents_for_reranking)
        # Return original documents as well, as no reranking happened
        return {"context": final_context, "documents": retrieved_documents}

    logger.info(f"Reranking {len(doc_contents_for_reranking)} documents for query: '{current_query}'")
    # Prepare pairs for the cross-encoder
    pairs = [[current_query, content] for content in doc_contents_for_reranking]
    try:
        # Get scores from the cross-encoder
        scores = reranker_model.predict(pairs)        
        logger.info(f"Reranking scores obtained (Top {app_config.RERANK_TOP_N}): {scores[:app_config.RERANK_TOP_N]}")
        # Combine docs with scores and sort
        scored_docs_content = list(zip(scores, doc_contents_for_reranking))
        scored_docs_content.sort(key=itemgetter(0), reverse=True)
        # Select top N documents based on config
        reranked_doc_contents = [content for score, content in scored_docs_content[:app_config.RERANK_TOP_N]]
        logger.info(f"Selected top {len(reranked_doc_contents)} documents after reranking.")
        # Concatenate the final context
        final_context = "\n\n".join(reranked_doc_contents)
        logger.info(f"Final reranked context snippet: {final_context[:1000]}...")
        # Update state with the final context
        # documents in state remains the larger pre-ranked set for now
        return {"context": final_context}
    except Exception as e:
        logger.error(f"Error during reranking: {e}. Using original context without reranking", exc_info=True)
        # Fallback: use the original concatenated context
        final_context = "\n\n".join(doc_contents_for_reranking)
        return {"context": final_context, "documents": retrieved_documents}  # Return original docs too
