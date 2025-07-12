import time
import torch
from typing import Dict, Any, List

from agi.memory.store import MemoryStore


class MemoryManager:
    """Memory management system for handling live and episodic memory operations"""

    def __init__(self, cfg: Dict[str, Any], emb_dim: int):
        """
        Initialize memory manager

        Args:
            cfg: Configuration dictionary containing memory_management settings
            emb_dim: Embedding dimension for vector storage
        """
        self.cfg = cfg
        self.emb_dim = emb_dim
        self.live_memory = None
        self.episodic_memory = None

        self._init_memory_stores()

    def _init_memory_stores(self):
        """Initialize live and episodic memory stores"""
        # Initialize live memory
        live_memory_cfg = self.cfg["memory_management"]["live_memory"]
        self.live_memory = MemoryStore(
            vector_size=self.emb_dim,  # Use embedding dimension as vector size
            cfg=live_memory_cfg,
            gpu=False,  # Force CPU to avoid CUDA issues
            reset=live_memory_cfg.get("reset", True),
            create=True,
        )

        # Initialize episodic memory
        episodic_memory_cfg = self.cfg["memory_management"]["episodic_memory"]
        self.episodic_memory = MemoryStore(
            vector_size=self.emb_dim,  # Use embedding dimension as vector size
            cfg=episodic_memory_cfg,
            gpu=False,  # Force CPU to avoid CUDA issues
            reset=episodic_memory_cfg.get("reset", True),
            create=True,
        )

    def process_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle memory operations for one processing step

        Args:
            context: Context dictionary containing state tensor and reward info

        Returns:
            Dictionary containing memory information and statistics
        """
        try:
            current_time = time.time()

            # Get state tensor (our simplified 47-dimensional state)
            state_tensor = context.get("state_tensor")
            
            # Get reward information from governor or motivator
            reward = 0.0
            emotion_reward = 0.0
            
            # Try to get reward from different sources
            if "governor_info" in context:
                reward = context["governor_info"].get("reward", 0.0)
                emotion_reward = context["governor_info"].get("simulated_reward", 0.0)
            elif "motivator_info" in context:
                # Get emotion guidance magnitude as emotion reward
                emotion_guidance = context["motivator_info"].get("emotion_guidance")
                if emotion_guidance is not None:
                    if isinstance(emotion_guidance, torch.Tensor):
                        emotion_reward = torch.norm(emotion_guidance).item()
                    else:
                        emotion_reward = float(emotion_guidance)

            # Store in live memory if we have a state tensor
            if state_tensor is not None:
                if isinstance(state_tensor, torch.Tensor):
                    state_vector = state_tensor.cpu().numpy().flatten()
                else:
                    state_vector = state_tensor
                
                # Ensure vector is the right size
                if len(state_vector) == self.emb_dim:
                    self.live_memory.memorize(
                        id=int(current_time * 1000),  # Use timestamp as ID
                        latent=state_vector.tolist(),
                        emotion_reward=emotion_reward,
                        reward=reward,
                    )

            # Periodic memory maintenance
            if int(current_time) % 60 == 0:  # Every minute
                self.live_memory.decay_on_retain_time()
                self.live_memory.decay_on_capacity("reward")

            memory_info = {
                "live_memory_size": self.live_memory.size,
                "episodic_memory_size": self.episodic_memory.size,
                "live_memory_stats": self.live_memory.get_reward_statistics(),
                "episodic_memory_stats": self.episodic_memory.get_reward_statistics(),
                "last_update": current_time,
            }

            return memory_info

        except Exception as e:
            print(f"Memory manager error: {e}")
            return {}

    def store_experience(self, experience: Dict[str, Any]) -> bool:
        """
        Store a complete experience in episodic memory

        Args:
            experience: Dictionary containing experience data

        Returns:
            True if successfully stored, False otherwise
        """
        try:
            if "latent" not in experience:
                return False

            current_time = time.time()

            self.episodic_memory.memorize(
                id=experience.get("id", int(current_time * 1000)),
                latent=experience["latent"],
                emotion_reward=experience.get("emotion_reward", 0.0),
                reward=experience.get("reward", 0.0),
            )

            return True

        except Exception as e:
            print(f"Store experience error: {e}")
            return False

    def retrieve_similar_memories(
        self, query_latent: torch.Tensor, k: int = 5, memory_type: str = "live",
        min_reward: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar memories based on latent similarity

        Args:
            query_latent: Query latent vector
            k: Number of similar memories to retrieve
            memory_type: Type of memory to search ("live" or "episodic")
            min_reward: Minimum reward threshold for retrieved memories

        Returns:
            List of similar memory entries
        """
        try:
            memory_store = (
                self.live_memory if memory_type == "live" else self.episodic_memory
            )

            if memory_store.size == 0:
                return []

            # Convert query to numpy if it's a tensor
            if isinstance(query_latent, torch.Tensor):
                query_vector = query_latent.cpu().numpy().flatten().tolist()
            else:
                query_vector = query_latent

            # Use reward-based recall if min_reward is specified
            if min_reward > 0:
                similar_memories = memory_store.recall_by_reward(
                    query_vector, k, min_reward
                )
            else:
                similar_memories = memory_store.recall(query_vector, k)

            # Convert to list of dictionaries
            memory_list = []
            for memory in similar_memories:
                memory_dict = {
                    "id": memory.id,
                    "score": memory.score,
                    "latent": memory.vector,
                    "emotion_reward": memory.payload.get("emotion_reward", 0.0),
                    "reward": memory.payload.get("reward", 0.0),
                    "timestamp": memory.payload.get("timestamp", 0.0),
                }
                memory_list.append(memory_dict)

            return memory_list

        except Exception as e:
            print(f"Retrieve memories error: {e}")
            return []

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics

        Returns:
            Dictionary containing memory statistics
        """
        try:
            stats = {
                "live_memory": {
                    "size": self.live_memory.size if self.live_memory else 0,
                    "capacity": getattr(self.live_memory, "capacity", "unknown"),
                    "reward_stats": self.live_memory.get_reward_statistics() if self.live_memory else {},
                },
                "episodic_memory": {
                    "size": self.episodic_memory.size if self.episodic_memory else 0,
                    "capacity": getattr(self.episodic_memory, "capacity", "unknown"),
                    "reward_stats": self.episodic_memory.get_reward_statistics() if self.episodic_memory else {},
                },
                "total_memories": (self.live_memory.size if self.live_memory else 0)
                + (self.episodic_memory.size if self.episodic_memory else 0),
            }

            return stats

        except Exception as e:
            print(f"Memory stats error: {e}")
            return {
                "live_memory": {"size": 0, "capacity": "unknown", "reward_stats": {}},
                "episodic_memory": {"size": 0, "capacity": "unknown", "reward_stats": {}},
                "total_memories": 0,
            }

    def consolidate_memories(self, min_reward: float = 0.5) -> bool:
        """
        Consolidate high-reward memories from live to episodic memory

        Args:
            min_reward: Minimum reward threshold for consolidation

        Returns:
            True if consolidation was successful, False otherwise
        """
        try:
            if not self.live_memory or self.live_memory.size == 0:
                return True

            # Get high-reward memories from live memory
            consolidated_memories, _ = self.live_memory.consolidate("reward")
            
            if consolidated_memories:
                # Store them in episodic memory
                episodic_points = []
                for memory in consolidated_memories:
                    if hasattr(memory, 'payload') and memory.payload.get("reward", 0) >= min_reward:
                        episodic_points.append(memory)
                
                if episodic_points:
                    self.episodic_memory.memorize_points(episodic_points)

            return True

        except Exception as e:
            print(f"Memory consolidation error: {e}")
            return False
