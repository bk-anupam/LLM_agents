import sqlite3
import time
from RAG_BOT.src.logger import logger

class UpdateManager:
    """
    UpdateManager handles tracking and cleanup of processed update IDs using a SQLite database.

    Attributes:
        db_path (str): Path to the SQLite database file.
        retention_period_seconds (int): Time in seconds to retain processed update records (default: 3 days).
        _conn (sqlite3.Connection): SQLite database connection object.

    """

    def __init__(self, db_path: str, retention_period_seconds: int = 259200): # 3 days
        self.db_path = db_path
        self.retention_period_seconds = retention_period_seconds
        self._conn = None
        self._ensure_connection()
        self._create_table()

    def _ensure_connection(self):
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
                logger.info("Successfully connected to SQLite database for UpdateManager.")
            except sqlite3.Error as e:
                logger.error(f"Error connecting to SQLite database: {e}", exc_info=True)
                raise

    def _create_table(self):
        try:
            with self._conn:
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS processed_updates (
                        update_id INTEGER PRIMARY KEY,
                        timestamp INTEGER NOT NULL
                    )
                """)
                logger.info("Table 'processed_updates' created or already exists.")
        except sqlite3.Error as e:
            logger.error(f"Error creating 'processed_updates' table: {e}", exc_info=True)

    def is_update_processed(self, update_id: int) -> bool:
        try:
            with self._conn:
                cursor = self._conn.execute("SELECT 1 FROM processed_updates WHERE update_id = ?", (update_id,))
                return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Error checking update_id {update_id}: {e}", exc_info=True)
            # In case of DB error, better to assume not processed to avoid missing a message.
            return False

    def mark_update_as_processed(self, update_id: int):
        try:
            current_timestamp = int(time.time())
            with self._conn:
                self._conn.execute(
                    "INSERT OR IGNORE INTO processed_updates (update_id, timestamp) VALUES (?, ?)",
                    (update_id, current_timestamp)
                )
        except sqlite3.Error as e:
            logger.error(f"Error marking update_id {update_id} as processed: {e}", exc_info=True)

    def cleanup_old_updates(self):
        try:
            cutoff_timestamp = int(time.time()) - self.retention_period_seconds
            with self._conn:
                cursor = self._conn.execute("DELETE FROM processed_updates WHERE timestamp < ?", (cutoff_timestamp,))
                logger.info(f"Cleaned up {cursor.rowcount} old update records.")
        except sqlite3.Error as e:
            logger.error(f"Error cleaning up old updates: {e}", exc_info=True)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("UpdateManager SQLite connection closed.")
