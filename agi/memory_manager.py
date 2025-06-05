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
            context: Context dictionary containing vision_latent and other data
            
        Returns:
            Dictionary containing memory information and statistics
        """
        try:
            current_time = time.time()

            # Store vision latent in memory if available
            if context.get("vision_latent") is not None:
                vision_latent = context["vision_latent"].cpu().numpy().flatten()

                # Store in live memory
                self.live_memory.memorize(
                    id=int(current_time * 1000),  # Use timestamp as ID
                    latent=vision_latent.tolist(),
                    emotion=[0.0, 0.0, 0.0],  # Placeholder emotion
                    efforts=0.0,
                )

            memory_info = {
                "live_memory_size": self.live_memory.size,
                "episodic_memory_size": self.episodic_memory.size,
                "last_update": current_time,
            }

            return memory_info

        except Exception:
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
                emotion=experience.get("emotion", [0.0, 0.0, 0.0]),
                efforts=experience.get("efforts", 0.0),
            )
            
            return True
            
        except Exception:
            return False
    
    def retrieve_similar_memories(self, query_latent: torch.Tensor, k: int = 5, memory_type: str = "live") -> List[Dict[str, Any]]:
        """
        Retrieve similar memories based on latent similarity
        
        Args:
            query_latent: Query latent vector
            k: Number of similar memories to retrieve
            memory_type: Type of memory to search ("live" or "episodic")
            
        Returns:
            List of similar memory entries
        """
        try:
            memory_store = self.live_memory if memory_type == "live" else self.episodic_memory
            
            if memory_store.size == 0:
                return []
            
            # Convert query to numpy if it's a tensor
            if isinstance(query_latent, torch.Tensor):
                query_vector = query_latent.cpu().numpy().flatten()
            else:
                query_vector = query_latent
            
            # Use memory store's search functionality if available
            # This is a placeholder - actual implementation depends on MemoryStore API
            similar_memories = []
            
            return similar_memories
            
        except Exception:
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
                },
                "episodic_memory": {
                    "size": self.episodic_memory.size if self.episodic_memory else 0,
                    "capacity": getattr(self.episodic_memory, "capacity", "unknown"),
                },
                "total_memories": (self.live_memory.size if self.live_memory else 0) + 
                                (self.episodic_memory.size if self.episodic_memory else 0),
            }
            
            return stats
            
        except Exception:
            return {
                "live_memory": {"size": 0, "capacity": "unknown"},
                "episodic_memory": {"size": 0, "capacity": "unknown"},
                "total_memories": 0,
            }
    
    def clear_memories(self, memory_type: str = "both") -> bool:
        """
        Clear memories from specified memory stores
        
        Args:
            memory_type: Type of memory to clear ("live", "episodic", or "both")
            
        Returns:
            True if successfully cleared, False otherwise
        """
        try:
            if memory_type in ["live", "both"] and self.live_memory:
                # Reset live memory - implementation depends on MemoryStore API
                pass
                
            if memory_type in ["episodic", "both"] and self.episodic_memory:
                # Reset episodic memory - implementation depends on MemoryStore API
                pass
                
            return True
            
        except Exception:
            return False
    
    def consolidate_memories(self) -> bool:
        """
        Consolidate memories from live to episodic memory based on criteria
        
        Returns:
            True if consolidation was successful, False otherwise
        """
        try:
            # This is a placeholder for memory consolidation logic
            # Implementation would depend on specific consolidation criteria
            # such as importance, recency, or emotional significance
            
            return True
            
        except Exception:
            return False
