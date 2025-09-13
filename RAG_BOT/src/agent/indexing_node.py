
import asyncio
from typing import Dict, Any
from RAG_BOT.src.agent.state import AgentState
from RAG_BOT.src.persistence.vector_store import VectorStore
from RAG_BOT.src.services.gcs_uploader import GCSUploaderService
from RAG_BOT.src.logger import logger
from RAG_BOT.src.config.config import Config


def get_indexing_and_sync_node(
        vector_store: VectorStore, 
        gcs_uploader: GCSUploaderService,
        config: Config
    ): 
    """
    Factory function to create the index_and_sync_node with dependencies.
    """
    async def _background_task(docs_to_index):
        """The actual indexing and syncing logic to run in the background."""
        try:
            logger.info(f"BG Task: Indexing {len(docs_to_index)} new document(s).")
            await asyncio.to_thread(
                vector_store.index_document, 
                docs_to_index, 
                semantic_chunk=config.SEMANTIC_CHUNKING
            )
            logger.info("BG Task: Successfully indexed new documents.")
            
            logger.info("BG Task: Triggering GCS sync for vector store.")
            await gcs_uploader.sync_vector_store()
            logger.info("BG Task: GCS sync completed.")
        except Exception as e:
            logger.error(f"Error during background indexing or GCS sync: {e}", exc_info=True)


    async def index_and_sync_node(state: AgentState) -> Dict[str, Any]:
        """
        Schedules background indexing and GCS sync for new documents.
        This node does NOT wait for the process to complete.
        """
        logger.info("---INDEX AND SYNC NODE---")
        docs_to_index = state.get("docs_to_index")

        if not docs_to_index:
            logger.info("No new documents to index. Skipping background task.")
            return {}

        logger.info(f"Scheduling background task to index {len(docs_to_index)} documents and sync with GCS.")
        # Create a background task that runs independently.
        # The state is immediately cleared to prevent re-processing.
        asyncio.create_task(_background_task(docs_to_index))
        
        # Return the state update to clear the documents list immediately.
        return {"docs_to_index": []}

    return index_and_sync_node
