from google.cloud import firestore
import time
from google.cloud.firestore_v1.base_query import FieldFilter
from RAG_BOT.src.logger import logger

class UpdateManager:
    """
    The UpdateManager class is a crucial component for making the Telegram bot robust and reliable. 
    Its primary purpose is to prevent the bot from processing the same message update more than once. 
    his is a common challenge with webhook-based bots, especially in serverless environments where a 
    "cold start" can cause the service to be slow to respond, leading Telegram to retry sending the 
    same update. This class ensures "idempotency" by keeping a record of every update ID it has already seen.

    Attributes:
        retention_period_seconds (int): Time in seconds to retain processed update records (default: 3 days).
    """

    def __init__(self, project_id: str, db_name: str = "rag-bot-firestore-db", retention_period_seconds: int = 259200): # 3 days
        self.db = firestore.Client(project=project_id, database=db_name)
        self.collection_ref = self.db.collection('processed_updates')
        self.retention_period_seconds = retention_period_seconds

    def is_update_processed(self, update_id: int) -> bool:
        """
        Checks if an update has been processed.

        Args:
            update_id (int): The update ID to check.

        Returns:
            bool: True if the update has been processed, False otherwise.
        """
        doc_ref = self.collection_ref.document(str(update_id))
        try:
            return doc_ref.get().exists
        except Exception as e:
            logger.error(f"Error checking update_id {update_id}: {e}", exc_info=True)
            # In case of DB error, better to assume not processed to avoid missing a message.
            return False

    def mark_update_as_processed(self, update_id: int):
        """
        Marks an update as processed.

        Args:
            update_id (int): The update ID to mark as processed.
        """
        doc_ref = self.collection_ref.document(str(update_id))
        try:
            current_timestamp = int(time.time())
            doc_ref.set({'timestamp': current_timestamp})
        except Exception as e:
            logger.error(f"Error marking update_id {update_id} as processed: {e}", exc_info=True)

    def cleanup_old_updates(self):
        """
        Deletes old update records from Firestore.
        """
        try:
            cutoff_timestamp = int(time.time()) - self.retention_period_seconds
            docs = self.collection_ref.where(filter=FieldFilter('timestamp', '<', cutoff_timestamp)).stream()
            deleted_count = 0
            for doc in docs:
                doc.reference.delete()
                deleted_count += 1
            logger.info(f"Cleaned up {deleted_count} old update records.")
        except Exception as e:
            logger.error(f"Error cleaning up old updates: {e}", exc_info=True)

    def close(self):
        """
        Closes the Firestore client.
        Note: Firestore clients are designed to be long-lived, so this may not be necessary.
        """
        pass