# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/retrieval_nodes.py
from operator import itemgetter
from typing import List, Optional
from sentence_transformers import CrossEncoder
from RAG_BOT.config import Config
from RAG_BOT.logger import logger
from RAG_BOT.agent.state import AgentState
from langchain_core.documents import Document


def rerank_context_node(state: AgentState, reranker_model: CrossEncoder, app_config: Config):
    """
    Reranks the initially retrieved documents based on the current query.
    Updates the state with the final, concatenated context string and the
    top N reranked documents, with scores added to their metadata.
    """
    logger.info("--- Executing Rerank Context Node ---")
    current_query = state.get('current_query')
    retrieved_documents: Optional[List[Document]] = state.get('documents')
    logger.info(f"Current query for reranking: '{current_query}'")

    if not retrieved_documents:
        logger.info("No documents found in state to rerank.")
        return {"context": "", "documents": []}

    # Filter for documents with valid content to avoid errors
    docs_to_rerank = [
        doc for doc in retrieved_documents
        if hasattr(doc, 'page_content') and isinstance(doc.page_content, str) and doc.page_content.strip()
    ]

    if not docs_to_rerank:
        logger.info("No document contents found for reranking after filtering.")
        # Return original docs as they are, with empty context
        return {"context": "", "documents": retrieved_documents}

    # Validate reranker and input
    if reranker_model is None:
        logger.warning("Reranker model not loaded. Using original document order.")
        # Take top N from the original list if no reranker
        top_docs = retrieved_documents[:app_config.RERANK_TOP_N]
        final_context = "\n\n".join([doc.page_content for doc in top_docs])
        return {"context": final_context, "documents": top_docs}

    logger.info(f"Reranking {len(docs_to_rerank)} documents for query: '{current_query}'")
    # Prepare pairs for the cross-encoder
    pairs = [[current_query, doc.page_content] for doc in docs_to_rerank]
    try:
        # Get scores from the cross-encoder. These are raw, unnormalized scores.
        scores = reranker_model.predict(pairs)

        # Combine the valid documents with their scores
        scored_docs = list(zip(scores, docs_to_rerank))
        # Sort documents based on scores in descending order
        scored_docs.sort(key=itemgetter(0), reverse=True)

        # Log all scores after sorting
        all_scores = [score for score, doc in scored_docs]
        logger.info(f"Reranking scores (sorted): {all_scores}")

        # Select top N documents and add score to metadata
        reranked_documents = []
        for score, doc in scored_docs[:app_config.RERANK_TOP_N]:
            # It's good practice to ensure the score is a standard Python float for serialization
            doc.metadata['rerank_score'] = float(score)
            reranked_documents.append(doc)

        logger.info(f"Selected top {len(reranked_documents)} documents after reranking.")

        # Concatenate the final context from the top N reranked documents
        final_context = "\n\n".join([doc.page_content for doc in reranked_documents])
        logger.info(f"Final reranked context snippet: {final_context[:1000]}...")

        # Update state with the final context and the sorted, scored, and trimmed documents
        return {"context": final_context, "documents": reranked_documents}

    except Exception as e:
        logger.error(f"Error during reranking: {e}. Using original context without reranking", exc_info=True)
        # Fallback: use the original concatenated context
        final_context = "\n\n".join(docs_to_rerank)
        return {"context": final_context, "documents": retrieved_documents}
