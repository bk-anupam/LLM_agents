from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Optional

from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from RAG_BOT.src.persistence.firestore_serializer import FirestoreSerializer


class AsyncFirestoreSaver(BaseCheckpointSaver):
    """An asynchronous checkpoint saver that stores checkpoints in Google Firestore."""

    checkpoints_collection_name: str = "checkpoints"
    writes_collection_name: str = "writes"

    def __init__(
        self,
        db: firestore.AsyncClient,
        *,
        serde: Optional[SerializerProtocol] = None,
    ):
        super().__init__(serde=serde)
        self.db = db
        self.firestore_serde = FirestoreSerializer(self.serde)
        self.checkpoints_collection = self.db.collection(self.checkpoints_collection_name)
        self.writes_collection = self.db.collection(self.writes_collection_name)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        
        if checkpoint_id := get_checkpoint_id(config):
            checkpoint_doc_ref = (
                self.checkpoints_collection
                .document(thread_id)
                .collection(self.checkpoints_collection_name)
                .document(checkpoint_id)
            )
            checkpoint_doc = await checkpoint_doc_ref.get()
        else:
            docs = (
                self.checkpoints_collection
                .document(thread_id)
                .collection(self.checkpoints_collection_name)
                .order_by("ts", direction=firestore.Query.DESCENDING).limit(1)
                .stream()
            )
            checkpoint_doc = None
            async for doc in docs:
                checkpoint_doc = doc
        
        if not checkpoint_doc or not checkpoint_doc.exists:
            return None

        checkpoint_data = checkpoint_doc.to_dict()
        
        # Fetch pending writes        
        writes_query = (
            self.writes_collection
            .document(thread_id)
            .collection(self.writes_collection_name)
            .where(filter=FieldFilter("checkpoint_id", "==", checkpoint_doc.id))
            .stream()
        )
        pending_writes = [            
            (
                write.to_dict()["task_id"],
                write.to_dict()["channel"],
                self.firestore_serde.loads_typed(
                    (write.to_dict()["type"], write.to_dict()["value"])
                ),
            )
            async for write in writes_query
        ]

        parent_config = self.firestore_serde.loads(checkpoint_data["parent_config"]) if checkpoint_data.get("parent_config") else None

        return CheckpointTuple(
            config,
            self.firestore_serde.loads_typed((checkpoint_data["type"], checkpoint_data["checkpoint"])),
            self.firestore_serde.loads(checkpoint_data["metadata"]),
            parent_config,
            pending_writes,
        )

    async def alist(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        query = (
            self.checkpoints_collection
            .document(thread_id)
            .collection(self.checkpoints_collection_name)
            .order_by("ts", direction=firestore.Query.DESCENDING)
        )

        if before:
            query = query.start_after({"ts": get_checkpoint_id(before)})

        if limit:
            query = query.limit(limit)
            
        async for doc in query.stream():
            doc_data = doc.to_dict()            
            writes_query = (
                self.writes_collection
                .document(thread_id)
                .collection(self.writes_collection_name)
                .where(filter=FieldFilter("checkpoint_id", "==", doc.id))
                .stream()
            )
            pending_writes = [                
                (
                    write.to_dict()["task_id"],
                    write.to_dict()["channel"],
                    self.firestore_serde.loads_typed(
                        (write.to_dict()["type"], write.to_dict()["value"])
                    ),
                )
                async for write in writes_query
            ]

            parent_config = self.firestore_serde.loads(doc_data["parent_config"]) if doc_data.get("parent_config") else None

            yield CheckpointTuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_id": doc.id}},
                self.firestore_serde.loads_typed((doc_data["type"], doc_data["checkpoint"])),
                self.firestore_serde.loads(doc_data["metadata"]),
                parent_config,
                pending_writes,
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        
        type_, serialized_checkpoint = self.firestore_serde.dumps_typed(checkpoint)
        serialized_metadata = self.firestore_serde.dumps(get_checkpoint_metadata(config, metadata))
        
        parent_config = config.get("parent_config")
        serialized_parent_config = self.firestore_serde.dumps(parent_config) if parent_config else None

        batch = self.db.batch()
        
        checkpoint_ref = (
            self.checkpoints_collection
            .document(thread_id)
            .collection(self.checkpoints_collection_name)
            .document(checkpoint["id"])
        )
        batch.set(checkpoint_ref, {
            "ts": checkpoint["ts"],
            "type": type_,
            "checkpoint": serialized_checkpoint,
            "metadata": serialized_metadata,
            "parent_config": serialized_parent_config,
        })
        
        await batch.commit()
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        
        batch = self.db.batch()
        for i, (channel, value) in enumerate(writes):
            type_, serialized_value = self.firestore_serde.dumps_typed(value)
            write_ref = (
                self.writes_collection
                .document(thread_id)
                .collection(self.writes_collection_name)
                .document(f"{checkpoint_id}_{task_id}_{i}")
            )
            batch.set(write_ref, {
                "checkpoint_id": checkpoint_id,
                # added task_id to uniquely identify writes
                "task_id": task_id,
                "channel": channel,
                "type": type_,
                "value": serialized_value,
            })
        await batch.commit()

    async def adelete_thread(self, thread_id: str) -> None:
        writes_ref = self.writes_collection.document(thread_id).collection(self.writes_collection_name)
        async for doc in writes_ref.stream():
            await doc.reference.delete()

        checkpoints_ref = self.checkpoints_collection.document(thread_id).collection(self.checkpoints_collection_name)
        async for doc in checkpoints_ref.stream():
            await doc.reference.delete()

        await self.checkpoints_collection.document(thread_id).delete()
        await self.writes_collection.document(thread_id).delete()
