from google.cloud import firestore
import google.cloud.exceptions
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

    def check_and_mark_update(self, update_id: int) -> bool:
        """
        Atomically checks if an update has been processed and marks it as processed.
        This prevents race conditions in concurrent environments.

        Args:
            update_id (int): The update ID to check and mark.

        Returns:
            bool: True if the update was already processed (it's a duplicate),
                  False if it was new and has now been marked.
        """
        doc_ref = self.collection_ref.document(str(update_id))
        try:
            # create() is an atomic operation. It will fail with an AlreadyExists
            # exception if the document already exists.
            current_timestamp = int(time.time())
            doc_ref.create({'timestamp': current_timestamp})            
            logger.info(f"First time seeing update_id {update_id}. Marking as processed.")
            return False 
        except google.cloud.exceptions.AlreadyExists:
            # The document already exists, so this is a duplicate update.            
            return True 
        except Exception as e:
            logger.error(f"Error checking and marking update_id {update_id}: {e}", exc_info=True)
            # In case of a different DB error, it's safer to assume it was NOT processed
            # to avoid missing a message. The next attempt might succeed.
            return False

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