from google.cloud import firestore
from RAG_BOT.src.logger import logger

class UserSettingsManager:
    """
    Manages user settings in a persistent Google Firestore database.
    """
    def __init__(self, project_id: str, db_name: str = "rag-bot-firestore-db"):
        """
        Initializes the UserSettingsManager.

        Args:
            project_id (str): The Google Cloud project ID.
            db_name (str): The name of the Firestore database.
        """
        self.db = firestore.Client(project=project_id, database=db_name)
        self.collection_ref = self.db.collection('user_preferences')

    
    async def get_user_settings(self, user_id: int) -> dict:
        """
        Retrieves settings for a given user.
        If the user does not exist, creates a default entry for them.

        Args:
            user_id (int): The user's unique ID.

        Returns:
            dict: A dictionary containing the user's settings (e.g., {'language_code': 'en', 'mode': 'default'}).
        """
        doc_ref = self.collection_ref.document(str(user_id))
        try:
            doc = doc_ref.get()
            if doc.exists:
                logger.debug(f"Found settings for user {user_id}: {doc.to_dict()}")
                return doc.to_dict()
            else:
                logger.info(f"No settings found for user {user_id}. Creating default entry.")
                default_settings = {'language_code': 'hi', 'mode': 'default'}
                doc_ref.set(default_settings)
                return default_settings
        except Exception as e:
            logger.error(f"Firestore error getting settings for user {user_id}: {e}", exc_info=True)
            # Fallback to in-memory default if DB fails
            return {'language_code': 'en', 'mode': 'default'}


    async def update_user_settings(self, user_id: int, language_code: str = None, mode: str = None):
        """
        Updates a user's settings in the database.

        Args:
            user_id (int): The user's unique ID.
            language_code (str, optional): The new language code. Defaults to None.
            mode (str, optional): The new mode. Defaults to None.
        """
        if language_code is None and mode is None:
            return  # Nothing to update

        doc_ref = self.collection_ref.document(str(user_id))
        updates = {}
        if language_code:
            updates['language_code'] = language_code
        if mode:
            updates['mode'] = mode

        try:
            doc_ref.set(updates, merge=True)
            logger.info(f"Successfully updated settings for user {user_id}: {updates}")
        except Exception as e:
            logger.error(f"Firestore error updating settings for user {user_id}: {e}", exc_info=True)
            raise

    
    def close_connection(self):
        """
        Closes the Firestore client connection.
        Note: Firestore clients are designed to be long-lived, so this may not be necessary.
        """
        pass