from collections import Counter, defaultdict
from langchain_chroma import Chroma
from typing import Optional, Dict, Any, List, Tuple 
from RAG_BOT.src.logger import logger
from RAG_BOT.src.config.config import Config

class ResultProcessor:
    """Handles result combination and Murli reconstruction."""
    
    @staticmethod
    def combine_and_deduplicate(
        semantic_docs: List[Any], 
        bm25_items: List[Tuple[str, Dict[str, Any]]]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Combine and deduplicate results from semantic and BM25 search."""
        combined_map = {}
        
        # Process semantic results
        for doc in semantic_docs:
            if hasattr(doc, 'page_content') and doc.page_content not in combined_map:
                metadata = getattr(doc, 'metadata', {})
                combined_map[doc.page_content] = (doc.page_content, metadata)
        
        # Process BM25 results
        for content, metadata in bm25_items:
            if isinstance(content, str) and content not in combined_map:
                combined_map[content] = (content, metadata)
        
        unique_items = list(combined_map.values())
        logger.info(f"Combined and deduplicated results: {len(unique_items)} unique items.")
        return unique_items
    
    def reconstruct_murlis(
        self,
        chunk_metadatas: List[Dict[str, Any]],
        vectordb: Chroma,
        config: Config
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Reconstruct full Murlis from chunks."""
        if not chunk_metadatas:
            logger.info("No chunk metadata provided for Murli reconstruction.")
            return []
        
        murli_identifiers = self._extract_murli_identifiers(chunk_metadatas)
        logger.info(f"Extracted {murli_identifiers} Murli identifiers from metadata.")
        if not murli_identifiers:
            return []
        
        sorted_murlis = self._get_sorted_murlis_by_relevance(murli_identifiers, config.MAX_RECON_MURLIS)
        logger.info(f"Identified {len(sorted_murlis)} distinct Murlis for reconstruction with metadata: {sorted_murlis}")
        
        return self._reconstruct_murlis_batch(sorted_murlis, vectordb, config)
    
    @staticmethod
    def _extract_murli_identifiers(metadatas: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Extract valid (date, language) identifiers from metadata."""
        return [
            (meta["date"], meta["language"])
            for meta in metadatas
            if meta.get("date") and meta.get("language")
        ]
    
    @staticmethod
    def _get_sorted_murlis_by_relevance(identifiers: List[Tuple[str, str]], max_murlis: int) -> List[Tuple[str, str]]:
        """Sort Murlis by relevance (chunk count) and limit to max."""
        murli_counts = Counter(identifiers)
        return [item[0] for item in murli_counts.most_common(max_murlis)]
    
    def _reconstruct_murlis_batch(
        self,
        murli_list: List[Tuple[str, str]],
        vectordb: Chroma,
        config: Config
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Reconstruct a batch of Murlis."""
        reconstructed = []
        
        for date_val, lang_val in murli_list:
            murli_result = self._reconstruct_single_murli(date_val, lang_val, vectordb, config)
            if murli_result:
                reconstructed.append(murli_result)
        
        return reconstructed
    
    def _reconstruct_single_murli(
        self,
        date_val: str,
        lang_val: str,
        vectordb: Chroma,
        config: Config
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Reconstruct a single Murli from its chunks."""
        logger.info(f"Reconstructing Murli for Date: {date_val}, Language: {lang_val}")
        
        try:
            chunks_data = vectordb.get(
                where={"$and": [{"date": date_val}, {"language": lang_val}]},
                include=["documents", "metadatas"],
                limit=config.MAX_CHUNKS_PER_MURLI_RECON
            )
            
            chunk_items = self._process_chunks_data(chunks_data, date_val, lang_val)
            if not chunk_items:
                return None
            
            return self._assemble_murli(chunk_items)
            
        except Exception as e:
            logger.error(f"Error reconstructing Murli for Date: {date_val}, Language: {lang_val}: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _process_chunks_data(chunks_data: Dict, date_val: str, lang_val: str) -> List[Dict]:
        """Process and validate chunk data."""
        if not chunks_data or not chunks_data.get('ids'):
            logger.warning(f"No chunks found for Murli Date: {date_val}, Language: {lang_val}")
            return []
        
        chunk_items = []
        documents = chunks_data.get('documents', [])
        metadatas = chunks_data.get('metadatas', [])
        
        for content, meta in zip(documents, metadatas):
            if content and isinstance(content, str) and meta and 'seq_no' in meta:
                chunk_items.append({
                    'content': content,
                    'metadata': meta,
                    'seq_no': meta['seq_no']
                })
        
        if not chunk_items:
            logger.warning(f"No valid chunks with seq_no found for Date: {date_val}, Language: {lang_val}")
        
        return chunk_items
    
    @staticmethod
    def _assemble_murli(chunk_items: List[Dict]) -> Tuple[str, Dict[str, Any]]:
        """Assemble chunks into a complete Murli."""
        chunk_items.sort(key=lambda x: x['seq_no'])
        
        full_content = "\n\n".join(item['content'] for item in chunk_items)
        
        # Use first chunk's metadata as representative
        representative_metadata = chunk_items[0]['metadata'].copy()
        representative_metadata.pop('seq_no', None)
        representative_metadata['reconstructed'] = True
        
        logger.debug(f"Successfully reconstructed context from {len(chunk_items)} chunks.")
        return (full_content, representative_metadata)


    def _extract_murli_identifier(self, meta: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Extracts a single (date, language) identifier from metadata."""
        if meta.get("date") and meta.get("language"):
            return (meta["date"], meta["language"])
        return None


    def reconstruct_from_sentence_windows(
        self,
        chunk_metadatas: List[Dict[str, Any]],
        vectordb: Chroma,
        config: Config
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Reconstructs a context window around EACH retrieved chunk.
        For each chunk in the input, it fetches a window of surrounding chunks
        (e.g., 2 before and 2 after) from the same document and assembles them
        into a single, larger context string.
        """
        if not chunk_metadatas:
            logger.info("No chunk metadata provided for sentence window reconstruction.")
            return []

        window_size = config.SENTENCE_WINDOW_SIZE
        logger.info(f"Reconstructing context using sentence window size: {window_size}")
        
        # Dictionary of dictionaries where outer key is (date, language) and inner key is seq_no, inner value is retrieval_type
        murli_retrieval_info = defaultdict(dict)
        for meta in chunk_metadatas:
            identifier = self._extract_murli_identifier(meta)
            if identifier and 'seq_no' in meta:
                # Map seq_no to its retrieval_type for this Murli
                murli_retrieval_info[identifier][meta['seq_no']] = meta.get('retrieval_type')

        # Pre-fetch all chunks for each relevant Murli to avoid multiple DB calls
        all_murli_chunks_map = {}
        for identifier in murli_retrieval_info.keys():
            date_val, lang_val = identifier
            try:
                # This is a single DB call per Murli
                chunks_data = vectordb.get(
                    where={"$and": [{"date": date_val}, {"language": lang_val}]},
                    include=["documents", "metadatas"],
                    limit=config.MAX_CHUNKS_PER_MURLI_RECON # Safeguard
                )
                processed_chunks = self._process_chunks_data(chunks_data, date_val, lang_val)
                if processed_chunks:
                    all_murli_chunks_map[identifier] = processed_chunks
            except Exception as e:
                logger.error(f"Failed to pre-fetch chunks for Murli {identifier}: {e}", exc_info=True)

        reconstructed_contexts = []
        for identifier, retrieved_seq_info in murli_retrieval_info.items():
            murli_total_chunks = all_murli_chunks_map.get(identifier)
            if not murli_total_chunks:
                logger.warning(f"No chunks available in map for Murli {identifier}. Skipping.")
                continue

            # For each originally retrieved chunk, build its window from the pre-fetched data
            for seq_no, retrieval_type in retrieved_seq_info.items():
                min_seq_window = max(1, seq_no - window_size)
                max_seq_window = seq_no + window_size
                
                logger.debug(f"Creating window for Murli {identifier}, center seq_no: {seq_no}, range: [{min_seq_window}, {max_seq_window}]")

                # Filter the pre-fetched chunks for this specific window
                window_chunks = [
                    chunk for chunk in murli_total_chunks 
                    if min_seq_window <= chunk['seq_no'] <= max_seq_window
                ]

                if not window_chunks:
                    continue

                # Assemble this specific window into a single context string
                assembled_context = self._assemble_murli(window_chunks)

                if assembled_context:
                    # The retrieval_type for the center chunk was preserved and is available
                    # in the 'retrieval_type' variable from our loop.
                    assembled_context = (assembled_context[0], {**assembled_context[1], 'retrieval_type': retrieval_type})
                    reconstructed_contexts.append(assembled_context)

        # Sort contexts by length (descending) to prioritize larger, more comprehensive contexts.
        reconstructed_contexts.sort(key=lambda item: len(item[0]), reverse=True)

        # Deduplicate contexts, discarding any that are fully contained within a larger one.        
        final_unique_contexts = []
        for content, meta in reconstructed_contexts:
            is_substring = False
            # Check if the current content is a substring of any already added (and longer) context.
            for existing_content, _ in final_unique_contexts:
                if content in existing_content:
                    is_substring = True
                    logger.info(f"Discarding redundant context of length {len(content)}, metadata: {meta}, "
                                f"as as it is contained in a larger context.")
                    break
            if not is_substring:
                final_unique_contexts.append((content, meta))

        logger.info(f"Reconstructed {len(final_unique_contexts)} unique context windows from {len(chunk_metadatas)} initial chunks.")
        return final_unique_contexts