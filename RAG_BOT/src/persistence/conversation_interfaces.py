from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ThreadInfo:
    """A dataclass to hold information about a conversation thread."""
    thread_id: str
    title: str
    created_at: datetime
    last_modified_at: datetime
    is_active: bool
    is_archived: bool
    summary_count: int

class AbstractThreadManager(ABC):
    """
    An abstract base class defining the interface for managing conversation threads.
    This allows for different backend implementations (e.g., Firestore, SQLite).
    """

    @abstractmethod
    async def get_active_thread_id(self, user_id: str) -> Optional[str]:
        """
        Gets the active thread ID for a user. If none exists, it should
        create a new one and return its ID.
        """
        pass

    @abstractmethod
    async def create_new_thread(self, user_id: str, title: Optional[str] = None) -> str:
        """
        Creates a new thread for the user, sets it as the active thread,
        and returns the new thread's ID. If a title is not provided, a
        default one should be generated.
        """
        pass

    @abstractmethod
    async def increment_summary_count(self, user_id: str, thread_id: str) -> int:
        """
        Increments the summary counter for a specific thread and returns
        the new count.
        """
        pass

    @abstractmethod
    async def get_summary_count(self, user_id: str, thread_id: str) -> int:
        """
        Retrieves the current summary count for a specific thread.
        """
        pass

    @abstractmethod
    async def list_threads(self, user_id: str) -> List[ThreadInfo]:
        """
        Returns a list of all conversation threads for a user, typically
        sorted by last modified date.
        """
        pass

    @abstractmethod
    async def switch_active_thread(self, user_id: str, thread_id: str) -> None:
        """
        Switches the active thread for a user to the specified thread_id.
        """
        pass

    @abstractmethod
    async def archive_thread(self, user_id: str, thread_id: str) -> None:
        """
        Marks a thread as archived, effectively closing it to new messages.
        """
        pass
