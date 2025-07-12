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
from src.utilities.vectordb.vec_store import VectorStore


class MemoryStore:
    strong_emotion_intensity = 0.5  # Threshold for strong emotions (scalar)

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
        emotion_reward: float,  # Simplified to scalar
        reward: float,         # Agent's reward
    ) -> None:
        self.memory_store.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=id,
                    vector=latent,
                    payload={
                        "emotion_reward": emotion_reward,
                        "reward": reward,
                        "emotion_intensity": abs(emotion_reward),  # Use absolute value as intensity
                        "timestamp": time.time(),
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

    def recall_by_reward(
        self,
        latent,
        k,
        min_reward: float = 0.0,
    ) -> List[ScoredPoint]:
        """Recall memories with minimum reward threshold"""
        recalled = self.memory_store.client.search(
            collection_name=self.collection_name,
            query_vector=latent,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="reward",
                        range=Range(gte=min_reward),
                    )
                ]
            ),
            limit=k,
        )

        return recalled

    def recall_all(
        self,
        payloads=["emotion_reward", "reward"],
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
            "reward": 0.5,  # Threshold for good experiences
        }

        try:
            consolidated = self.memory_store.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key=attr,
                            range=Range(gt=thresh_lookup.get(attr, 0.0)),
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

    def get_reward_statistics(self) -> dict:
        """Get statistics about stored rewards"""
        if self.size == 0:
            return {"count": 0, "avg_reward": 0.0, "max_reward": 0.0, "min_reward": 0.0}
        
        rewards = self.memory_attr("reward")
        return {
            "count": len(rewards),
            "avg_reward": np.mean(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
        }


if __name__ == "__main__":
    cfg = {
        "host": "localhost",
        "port": 6333,
        "collection": "test_simplified",
        "gpu": False,
        "capacity": 100,
        "retain_time": 3600,  # 1 hour
    }

    memory = MemoryStore(vector_size=47, cfg=cfg)  # Updated to match our simplified state tensor
    vector = np.random.rand(47)
    
    # Test with simplified scalar values
    memory.memorize(
        id=0, 
        latent=vector, 
        emotion_reward=0.8,  # Positive emotion
        reward=0.5           # Moderate reward
    )
    
    print("Memory stored successfully!")
    print(f"Memory statistics: {memory.get_reward_statistics()}")
