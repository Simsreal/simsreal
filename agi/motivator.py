import torch
from typing import Dict, Any
from importlib import import_module


class Motivator:
    """Motivator system for handling intrinsics and emotion guidance"""

    def __init__(
        self,
        cfg: Dict[str, Any],
        device: torch.device,
        emb_dim: int,
        live_memory=None,
        episodic_memory=None,
    ):
        """
        Initialize motivator system

        Args:
            cfg: Configuration dictionary containing intrinsics settings
            device: PyTorch device for tensor operations
            emb_dim: Embedding dimension for latent representations
            live_memory: Live memory store reference
            episodic_memory: Episodic memory store reference
        """
        self.cfg = cfg
        self.device = device
        self.emb_dim = emb_dim
        self.live_memory = live_memory
        self.episodic_memory = episodic_memory

        # Initialize motivator components
        self._init_intrinsics()

    def _init_intrinsics(self):
        """Initialize intrinsics system"""
        try:
            # Load intrinsics module
            intrinsics_module = import_module("agi.intrinsics")
            intrinsic_lookup = {
                name: getattr(intrinsics_module, name)
                for name in intrinsics_module.__all__
            }

            # Get intrinsics from config
            self.intrinsics = self.cfg.get("intrinsics", [])

            # Create intrinsic indices
            self.intrinsic_indices = {
                intrinsic: idx for idx, intrinsic in enumerate(self.intrinsics)
            }

            # Initialize intrinsic motivators
            self.motivators = {}
            for intrinsic in self.intrinsics:
                if intrinsic in intrinsic_lookup:
                    self.motivators[intrinsic] = intrinsic_lookup[intrinsic](
                        id=self.intrinsic_indices[intrinsic],
                        live_memory_store=self.live_memory,
                        episodic_memory_store=self.episodic_memory,
                    )

            # Initialize emotion guidance tracking
            self.current_emotion_guidance = None

        except Exception:
            self.intrinsics = []
            self.motivators = {}
            self.intrinsic_indices = {}
            self.current_emotion_guidance = None

    def process_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate emotion guidance from intrinsics"""
        try:
            if not self.motivators:
                return {"emotion_guidance": None}

            # Get vision latent as primary latent representation
            latent = context.get("vision_latent")

            # If no vision latent, create a simple representation from raycast data
            if latent is None:
                # Create a simple latent from raycast distances (fallback)
                distances = context.get("raycast_info", {}).get("distances", [])
                if distances:
                    # Pad or truncate to fixed size and normalize
                    padded_distances = (distances + [100.0] * self.emb_dim)[
                        : self.emb_dim
                    ]
                    latent = (
                        torch.tensor(padded_distances, dtype=torch.float32).to(
                            self.device
                        )
                        / 100.0
                    )
                    latent = latent.unsqueeze(0)  # Add batch dimension
                else:
                    return {"emotion_guidance": None}

            # Create dummy emotion (neutral state)
            dummy_emotion = torch.zeros(3, dtype=torch.float32).to(
                self.device
            )  # PAD format [P, A, D]

            # Create dummy governance (equal weights for all intrinsics)
            dummy_governance = torch.ones(
                len(self.intrinsics), dtype=torch.float32
            ) / len(self.intrinsics)
            dummy_governance = dummy_governance.to(self.device)

            # Extract agent state and other important info
            agent_state = context.get("agent_state", 0)
            hit_point = context.get("hit_point", 100)
            hunger = context.get("hunger", 0.0)
            detection_summary = context.get("detection_summary", {})

            # Create information dictionary for intrinsics
            information = {
                "latent": latent.flatten(),
                "emotion": dummy_emotion,
                "governance": dummy_governance,
                "agent_state": agent_state,
                "hit_point": hit_point,
                "hunger": hunger,
                "detection_summary": detection_summary,
                "robot_state": context.get("raw_perception", {}),
                "force_on_geoms": torch.zeros(1, dtype=torch.float32).to(
                    self.device
                ),  # Backward compatibility
            }

            # Collect emotion guidance from all intrinsics
            emotion_guidances = []
            for intrinsic_name in self.intrinsics:
                if intrinsic_name in self.motivators:
                    intrinsic = self.motivators[intrinsic_name]

                    try:
                        # Run the intrinsic logic
                        intrinsic.impl(
                            information, {}, None
                        )  # No brain_shm, no physics

                        # Try to get emotion guidance from the priority queue
                        guidance = intrinsic.priorities["emotion"].get_nowait()[1]
                        emotion_guidances.append(guidance)
                    except Exception:
                        continue  # No guidance available from this intrinsic

            # Average emotion guidance if any exists
            if emotion_guidances:
                combined_emotion_guidance = torch.stack(emotion_guidances).mean(dim=0)
                self.current_emotion_guidance = combined_emotion_guidance
            else:
                self.current_emotion_guidance = None

            return {
                "emotion_guidance": self.current_emotion_guidance,
                "active_intrinsics": len(emotion_guidances),
                "total_intrinsics": len(self.motivators),
                "agent_state": agent_state,
                "threats_detected": detection_summary.get("trap", {}).get("count", 0)
                + detection_summary.get("enemy", {}).get("count", 0),
                "goals_detected": detection_summary.get("goal", {}).get("count", 0)
                + detection_summary.get("food", {}).get("count", 0),
            }

        except Exception:
            return {"emotion_guidance": None}
