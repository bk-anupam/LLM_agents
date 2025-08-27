import uuid
from datetime import datetime, timezone, MINYEAR
from typing import List, Optional
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from RAG_BOT.src.persistence.conversation_interfaces import AbstractThreadManager, ThreadInfo
from RAG_BOT.src.logger import logger

class FirestoreThreadManager(AbstractThreadManager):
    """
    An implementation of AbstractThreadManager that uses Google Firestore
    with a dedicated top-level 'threads' collection.
    """
    def __init__(
        self,
        project_id: str,
        db_name: str = "rag-bot-firestore-db",
        user_collection_name: str = "user_preferences",
        thread_collection_name: str = "threads"
    ):
        self.client = firestore.AsyncClient(project=project_id, database=db_name)
        self.user_collection = self.client.collection(user_collection_name)
        self.thread_collection = self.client.collection(thread_collection_name)


    async def get_active_thread_id(self, user_id: str) -> Optional[str]:
        """Gets the active thread ID for a user. Creates one if none exist."""
        user_doc_ref = self.user_collection.document(str(user_id))
        user_doc = await user_doc_ref.get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            active_id = user_data.get("active_thread_id")
            if active_id:
                return active_id

        # If document doesn't exist, or active_thread_id is missing, create a new one with a UTC-based title.
        # Let create_new_thread handle the title generation for consistency.
        return await self.create_new_thread(user_id)


    async def create_new_thread(self, user_id: str, title: Optional[str] = None) -> str:
        """
        Creates a new thread in the 'threads' collection, sets it as active
        for the user, and returns its ID. Uses a batched write for atomicity.
        If title is not provided, a default one is generated.
        """
        thread_id = f"thread_{uuid.uuid4()}"
        now = datetime.now(timezone.utc)

        # If no title is provided, create a default one using the 'now' timestamp.
        if title is None:
            title = f"Chat from {now.strftime('%Y-%m-%d %H:%M')} UTC"
        # The full thread data, including the user_id for querying.
        new_thread_data = {
            "user_id": str(user_id),
            "title": title,
            "created_at": now,
            "last_modified_at": now,
            "is_archived": False,
            "summary_count": 0,
        }

        # Use a batched write to ensure both operations succeed or fail together.
        batch = self.client.batch()
        # 1. Create the new thread document in the 'threads' collection.
        thread_doc_ref = self.thread_collection.document(thread_id)
        batch.set(thread_doc_ref, new_thread_data)
        # 2. Update the user's document to set the new active thread ID.
        user_doc_ref = self.user_collection.document(str(user_id))
        batch.set(user_doc_ref, {"active_thread_id": thread_id}, merge=True)
        await batch.commit()
        logger.info(f"Created new thread {thread_id} for user {user_id} with title '{title}'")
        return thread_id


    async def increment_summary_count(self, user_id: str, thread_id: str) -> int:
        """Increments and returns the summary count for a specific thread."""
        # This now operates directly on the thread's document.
        thread_doc_ref = self.thread_collection.document(thread_id)
        await thread_doc_ref.update({"summary_count": firestore.Increment(1)})
        # To return the new value, we must read it back.
        return await self.get_summary_count(user_id, thread_id)


    async def get_summary_count(self, user_id: str, thread_id: str) -> int:
        """Retrieves the summary count for a specific thread."""
        # This now reads directly from the thread's document.
        # We also fetch the user_id to verify ownership.
        thread_doc_ref = self.thread_collection.document(thread_id)
        thread_doc = await thread_doc_ref.get(field_paths=["summary_count", "user_id"])
        if thread_doc.exists:
            data = thread_doc.to_dict()
            # Verify the user requesting the count is the owner of the thread.
            if data.get("user_id") == str(user_id):
                return data.get("summary_count") or 0
        return 0


    async def list_threads(self, user_id: str) -> List[ThreadInfo]:
        """Lists all threads for a user by querying the 'threads' collection."""
        # First, get the active thread ID from the user's document.
        user_doc = await self.user_collection.document(str(user_id)).get()
        active_thread_id = user_doc.to_dict().get("active_thread_id") if user_doc.exists else None

        # Now, query the 'threads' collection for all threads by this user.
        # NOTE: This requires a composite index in Firestore.
        query = self.thread_collection.where(
            filter=FieldFilter("user_id", "==", str(user_id))
        ).order_by("last_modified_at", direction=firestore.Query.DESCENDING)

        thread_list = []
        async for doc in query.stream():
            tdata = doc.to_dict()
            tid = doc.id
            thread_list.append(
                ThreadInfo(
                    thread_id=tid,
                    title=tdata.get("title", ""),
                    created_at=tdata.get("created_at"),
                    last_modified_at=tdata.get("last_modified_at"),
                    is_active=(tid == active_thread_id),
                    is_archived=tdata.get("is_archived", False),
                    summary_count=tdata.get("summary_count", 0),
                )
            )
        return thread_list


    async def switch_active_thread(self, user_id: str, thread_id: str) -> None:
        """Switches the user's active thread by updating the user document."""
        user_doc_ref = self.user_collection.document(str(user_id))
        await user_doc_ref.update({"active_thread_id": thread_id})


    async def archive_thread(self, user_id: str, thread_id: str) -> None:
        """Marks a thread as archived by updating its document directly."""
        thread_doc_ref = self.thread_collection.document(thread_id)
        await thread_doc_ref.update({"is_archived": True})
