import time
from typing import Any, List, Sequence, Tuple

import numpy as np
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    PointStruct,
    Range,
    Record,
    ScoredPoint,
    UpdateResult,
)
from loguru import logger
from src.utilities.emotion.pad import get_emotion_magnitude
from src.utilities.vectordb.vec_store import VectorStore


class MemoryStore:
    strong_emotion_intensity = -1

    def __init__(
        self,
        vector_size,
        cfg,
        gpu: bool = False,
        reset=True,
        create=True,
    ):
        self.vector_size = vector_size
        self.capacity = cfg["capacity"]
        self.collection_name = cfg["collection"]
        self.retain_time = cfg["retain_time"]
        self.host = cfg["host"]
        self.port = cfg["port"]
        self.gpu = gpu

        self.memory_store = VectorStore(
            host=cfg["host"],
            port=cfg["port"],
            vector_size=self.vector_size,
            collection_name=self.collection_name,
            gpu=self.gpu,
            reset=reset,
            create=create,
        )

    @property
    def size(self) -> int:
        return self.memory_store.client.count(
            self.collection_name,
            exact=True,
        ).count

    def memory_attr(
        self,
        attr: str,
    ) -> List[float]:
        result = self.memory_store.client.scroll(
            collection_name=self.collection_name,
            limit=self.size,
            with_payload=[attr],
            with_vectors=False,
        )
        records = result[0]
        attr_values = [
            record.payload[attr] for record in records if record.payload is not None
        ]
        return attr_values

    def memorize(
        self,
        id,
        latent,
        emotion,
        efforts,
    ) -> None:
        self.memory_store.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=id,
                    vector=latent,
                    payload={
                        "emotion": emotion,
                        "emotion_intensity": get_emotion_magnitude(emotion),
                        "timestamp": time.time(),
                        "efforts": efforts,
                    },
                )
            ],
        )

    def memorize_points(self, points: List[PointStruct]) -> None:
        if len(points) == 0:
            return

        pointstructs = []
        for point in points:
            pointstructs.append(
                PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload,
                )
            )
        self.memory_store.client.upsert(
            collection_name=self.collection_name,
            points=pointstructs,
        )

    def update_points(self) -> None:
        raise NotImplementedError

    def recall(
        self,
        latent,
        k,
    ) -> List[ScoredPoint]:
        recalled = self.memory_store.client.search(
            collection_name=self.collection_name,
            query_vector=latent,
            limit=k,
        )

        return recalled

    def recall_all(
        self,
        payloads=["emotion"],
    ) -> List[Record]:
        memory = self.memory_store.client.scroll(
            collection_name=self.collection_name,
            limit=self.size,
            with_payload=payloads,
            with_vectors=True,
        )
        return memory[0]

    def consolidate(self, attr) -> Tuple[Sequence[PointStruct | Record], Any]:
        thresh_lookup = {
            "emotion_intensity": self.strong_emotion_intensity,
        }

        try:
            consolidated = self.memory_store.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key=attr,
                            range=Range(gt=thresh_lookup[attr]),
                        )
                    ]
                ),
                limit=self.size,
                with_vectors=True,
                with_payload=True,
            )
        except Exception as e:
            logger.warning(f"error in consolidate: {e}")
            return ([], None)

        return consolidated

    def decay_on_retain_time(
        self,
    ) -> UpdateResult | None:
        if self.size == 0:
            return

        deletions = self.memory_store.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="timestamp",
                            range=Range(lt=time.time() - self.retain_time),
                        )
                    ]
                )
            ),
        )

        return deletions

    def decay_on_capacity(
        self,
        attr,
    ) -> None:
        size = self.size
        if size == 0:
            return

        if size < self.capacity:
            return

        attr_values = self.memory_attr(attr)
        attr_values.sort(reverse=True)
        attr_cutoff = attr_values[self.capacity - 1]

        self.memory_store.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key=attr,
                            range=Range(lt=attr_cutoff),
                        )
                    ]
                )
            ),
        )


if __name__ == "__main__":
    cfg = {
        "host": "localhost",
        "port": 6333,
        "collection": "test4",
        "gpu": False,
        "capacity": 2,
        "retain_time": 5,
    }

    memory = MemoryStore(vector_size=16, cfg=cfg)
    vector = np.random.rand(16)
    memory.memorize(0, vector, [0.3, 1, 1], [0.3, 1, 1])
