# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/retrieval_nodes.py
from collections import defaultdict
from operator import itemgetter
from typing import List, Optional
from sentence_transformers import CrossEncoder
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.logger import logger
from RAG_BOT.src.agent.state import AgentState
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
        return {"retrieved_context": "", "documents": []}

    # Filter for documents with valid content to avoid errors
    docs_to_rerank = [
        doc for doc in retrieved_documents
        if hasattr(doc, 'page_content') and isinstance(doc.page_content, str) and doc.page_content.strip()
    ]

    if not docs_to_rerank:
        logger.info("No document contents found for reranking after filtering.")
        # Return original docs as they are, with empty context
        return {"retrieved_context": "", "documents": retrieved_documents}

    # Validate reranker and input
    if reranker_model is None:
        logger.warning("Reranker model not loaded. Using original document order.")
        # Take top N from the original list if no reranker
        top_docs = retrieved_documents[:app_config.RERANK_TOP_N]
        final_context = "\n\n".join([doc.page_content for doc in top_docs])
        return {"retrieved_context": final_context, "documents": top_docs}

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
        final_context_parts = []
        
        mode = state.get('mode', 'default') # Get the mode from the state

        if mode == 'research':
            content_by_source = defaultdict(list)
            for score, doc in scored_docs[:app_config.RERANK_TOP_N]:
                doc.metadata['rerank_score'] = float(score)
                reranked_documents.append(doc)
                murli_date = doc.metadata.get('date', None)
                murli_language = doc.metadata.get('language', None)            
                if murli_date and murli_language:
                    source_key = f"Murli {murli_date}"
                    content_by_source[source_key].append(doc.page_content)
                else:
                    content_by_source["Miscellaneous Context"].append(doc.page_content)
            
            for source, contents in content_by_source.items():
                final_context_parts.append(f"Source: {source}\nContent:\n" + "\n\n".join(contents))
            final_context = "\n\n---\n\n".join(final_context_parts)
        else: # Default mode: plain concatenated text
            for score, doc in scored_docs[:app_config.RERANK_TOP_N]:
                doc.metadata['rerank_score'] = float(score)
                reranked_documents.append(doc)
                final_context_parts.append(doc.page_content)
            final_context = "\n\n".join(final_context_parts)

        logger.info(f"Selected top {len(reranked_documents)} documents after reranking.")
        logger.info(f"Final reranked context snippet (mode: {mode}): {final_context[:1000]}...")

        # Update state with the final context and the sorted, scored, and trimmed documents
        return {"retrieved_context": final_context, "documents": reranked_documents}

    except Exception as e:
        logger.error(f"Error during reranking: {e}. Using original context without reranking", exc_info=True)
        # Fallback: use the original concatenated context
        final_context = "\n\n".join([doc.page_content for doc in docs_to_rerank])
        return {"retrieved_context": final_context, "documents": docs_to_rerank}
