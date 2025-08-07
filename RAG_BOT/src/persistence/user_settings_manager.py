import sqlite3
import threading
from RAG_BOT.src.logger import logger

class UserSettingsManager:
    """
    Manages user settings in a persistent SQLite database.
    This class is designed to be thread-safe.
    """
    def __init__(self, db_path: str):
        """
        Initializes the UserSettingsManager.

        Args:
            db_path (str): The path to the SQLite database file.
        """
        self.db_path = db_path
        self._local = threading.local()  # For thread-local connection management
        self._create_table()


    def _get_connection(self):
        """
        Returns a thread-local database connection.
        """
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn


    def _create_table(self):
        """
        Creates the 'user_preferences' table if it doesn't exist.
        """
        try:
            conn = self._get_connection()
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        user_id INTEGER PRIMARY KEY,
                        language_code TEXT NOT NULL DEFAULT 'en',
                        mode TEXT NOT NULL DEFAULT 'default'
                    )
                """)
            logger.info("Successfully ensured 'user_preferences' table exists.")
        except sqlite3.Error as e:
            logger.error(f"Database error while creating 'user_preferences' table: {e}", exc_info=True)
            raise


    async def get_user_settings(self, user_id: int) -> dict:
        """
        Retrieves settings for a given user.
        If the user does not exist, creates a default entry for them.

        Args:
            user_id (int): The user's unique ID.

        Returns:
            dict: A dictionary containing the user's settings (e.g., {'language_code': 'en', 'mode': 'default'}).
        """
        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                cursor.execute("SELECT language_code, mode FROM user_preferences WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()

                if row:
                    logger.debug(f"Found settings for user {user_id}: {dict(row)}")
                    return dict(row)
                else:
                    logger.info(f"No settings found for user {user_id}. Creating default entry.")
                    # User not found, create with default values and return them
                    cursor.execute(
                        "INSERT INTO user_preferences (user_id, language_code, mode) VALUES (?, 'en', 'default')",
                        (user_id,)
                    )
                    return {'language_code': 'en', 'mode': 'default'}
        except sqlite3.Error as e:
            logger.error(f"Database error getting settings for user {user_id}: {e}", exc_info=True)
            # Fallback to in-memory default if DB fails
            return {'language_code': 'en', 'mode': 'default'}

    
    async def update_user_settings(self, user_id: int, language_code: str = None, mode: str = None):
        """
        Updates a user's settings in the database.
        Uses INSERT OR REPLACE to handle both new and existing users.

        Args:
            user_id (int): The user's unique ID.
            language_code (str, optional): The new language code. Defaults to None.
            mode (str, optional): The new mode. Defaults to None.
        """
        if language_code is None and mode is None:
            return # Nothing to update

        try:
            conn = self._get_connection()
            with conn:
                cursor = conn.cursor()
                # First, ensure the user exists with current or default settings
                cursor.execute(
                    "INSERT OR IGNORE INTO user_preferences (user_id) VALUES (?)",
                    (user_id,)
                )

                # Now, build and execute the update statement
                updates = []
                params = []
                if language_code:
                    updates.append("language_code = ?")
                    params.append(language_code)
                if mode:
                    updates.append("mode = ?")
                    params.append(mode)
                
                params.append(user_id)
                
                update_query = f"UPDATE user_preferences SET {', '.join(updates)} WHERE user_id = ?"
                
                cursor.execute(update_query, tuple(params))
                logger.info(f"Successfully updated settings for user {user_id}: lang={language_code}, mode={mode}")

        except sqlite3.Error as e:
            logger.error(f"Database error updating settings for user {user_id}: {e}", exc_info=True)
            raise


    def close_connection(self):
        """
        Closes the thread-local database connection if it exists.
        """
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            logger.info("Database connection closed for thread.")
