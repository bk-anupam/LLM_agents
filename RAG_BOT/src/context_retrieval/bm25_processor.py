from operator import itemgetter
from langchain_chroma import Chroma
from typing import Dict, Any, List, Tuple 
from RAG_BOT.src.logger import logger
from rank_bm25 import BM25Okapi

class BM25Processor:
    """Handles BM25 search operations."""
    
    def __init__(self, vectordb: Chroma):
        self.vectordb = vectordb
    
    def get_scoped_corpus(self, filter_dict: Dict[str, Any], max_docs: int = 500) -> List[Tuple[str, Dict[str, Any]]]:
        """Retrieve filtered corpus for BM25 search."""
        if not filter_dict:
            logger.info("No filters provided for BM25 corpus scoping.")
            return []
        
        logger.info(f"Fetching scoped corpus for BM25 with filters: {filter_dict}")
        
        try:
            results = self.vectordb.get(
                where=filter_dict,
                limit=max_docs,
                include=["documents", "metadatas"]
            )
            
            corpus_items = self._extract_corpus_items(results)
            logger.info(f"Fetched {len(corpus_items)} items for BM25 scoped corpus.")
            return corpus_items
            
        except Exception as e:
            logger.error(f"Error fetching scoped corpus for BM25: {e}", exc_info=True)
            return []
    
    @staticmethod
    def _extract_corpus_items(results: Dict) -> List[Tuple[str, Dict[str, Any]]]:
        """Extract valid corpus items from Chroma results."""
        if not results or not results.get('ids'):
            return []
        
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        
        return [
            (content, meta) 
            for content, meta in zip(documents, metadatas)
            if content and isinstance(content, str)
        ]
    
    @staticmethod
    def search(query: str, corpus_items: List[Tuple[str, Dict[str, Any]]], k: int) -> List[Tuple[str, Dict[str, Any]]]:
        """Perform BM25 search on corpus items."""
        if not corpus_items or not query:
            logger.info("BM25: Empty corpus or query. Skipping search.")
            return []
        
        logger.info(f"BM25: Performing search for query '{query}' on corpus of {len(corpus_items)} items.")
        
        try:
            corpus_contents = [item[0] for item in corpus_items]
            tokenized_corpus = [doc.lower().split() for doc in corpus_contents]
            tokenized_query = query.lower().split()
            
            bm25 = BM25Okapi(tokenized_corpus)
            doc_scores = bm25.get_scores(tokenized_query)
            
            scored_items = list(zip(doc_scores, corpus_items))
            scored_items.sort(key=itemgetter(0), reverse=True)
            
            results = []
            # Unpack the item tuple into content and metadata
            for score, (content, metadata) in scored_items[:k]:
                if score > 0:
                    # Add the retrieval_type to the metadata dictionary
                    metadata['retrieval_type'] = 'bm25'
                    results.append((content, metadata))
            
            logger.info(f"BM25: Retrieved {len(results)} documents with scores > 0 and added 'retrieval_type' metadata.")
            if results:
                logger.debug(f"BM25: Top {len(results)} results: {[item[0][:100] for item in results]}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during BM25 search: {e}", exc_info=True)
            return []