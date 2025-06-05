import torch
from typing import Dict, Any, List

from agi.learning.conscious.titans import Titans
from agi.learning.emotions import TitansAlphaSR


class Governor:
    """Governor system for decision making and movement control"""
    
    def __init__(self, cfg: Dict[str, Any], device: torch.device, emb_dim: int, intrinsics: List[str]):
        """
        Initialize governor system
        
        Args:
            cfg: Configuration dictionary containing brain settings
            device: PyTorch device for tensor operations
            emb_dim: Embedding dimension for latent representations
            intrinsics: List of intrinsic names for policy dimension calculation
        """
        self.cfg = cfg
        self.device = device
        self.emb_dim = emb_dim
        self.intrinsics = intrinsics
        
        # Initialize governor components
        self._init_governor()
    
    def _init_governor(self):
        """Initialize the governor (decision making system)"""
        try:
            # Movement symbols
            self.movement_symbols = [
                "moveforward",
                "movebackward",
                "lookleft",
                "lookright",
                "idle",
                "standup",
            ]
            movement_dim = len(self.movement_symbols)
            intrinsic_dim = len(self.intrinsics)
            total_policy_dim = intrinsic_dim + movement_dim

            # Brain configuration
            brain_cfg = self.cfg["brain"]
            self.ctx_len = brain_cfg["ctx_len"]

            # Initialize Titans model
            self.titans_model = Titans(
                self.emb_dim,  # latent_dim
                brain_cfg["titans"]["chunk_size"],
                self.device,
                total_policy_dim,
            ).to(self.device)

            # Initialize TitansAlphaSR
            self.titans_alphasr = TitansAlphaSR(
                self.titans_model, self.intrinsics, self.movement_symbols, self.device
            )

            # Initialize optimizer and context
            self.governor_optimizer = torch.optim.Adam(
                self.titans_model.parameters(), lr=0.001
            )
            self.ctx = torch.zeros(
                (1, self.ctx_len, self.emb_dim), dtype=torch.float32
            ).to(self.device)

            # Counters for MCTS maintenance
            self.governor_counter = 0

        except Exception:
            self.titans_model = None
            self.titans_alphasr = None
            self.movement_symbols = ["idle"]  # Fallback
    
    def fifo_context_update(self, ctx, x):
        """FIFO context update for maintaining temporal context"""
        x = x.to(self.device)
        return torch.cat((ctx[:, 1:, :], x.unsqueeze(1)), dim=1)
    
    def process_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate movement decisions using the governor"""
        try:
            if self.titans_alphasr is None:
                return {"movement_command": "idle", "decision_confidence": 0.0}

            # Get latent representation
            latent = context.get("vision_latent")
            emotion_guidance = context.get("emotion_guidance")

            # Ensure emotion_guidance is on the correct device
            if emotion_guidance is not None:
                if isinstance(emotion_guidance, torch.Tensor):
                    emotion_guidance = emotion_guidance.to(self.device)

            # If no vision latent, create one from raycast data
            if latent is None:
                distances = context.get("raycast_info", {}).get("distances", [])
                if distances:
                    # Create normalized latent from raycast distances
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
                    # Use zero latent as fallback
                    latent = torch.zeros(1, self.emb_dim, dtype=torch.float32).to(
                        self.device
                    )
            else:
                # Ensure vision latent is on the correct device
                latent = latent.to(self.device)

            # Update context with FIFO
            self.ctx = self.fifo_context_update(self.ctx, latent)
            state = latent.flatten()

            # Get outputs from TitansAlphaSR with emotion guidance
            outputs = self.titans_alphasr.forward(
                state,
                self.ctx,
                emotion_guidance=emotion_guidance,
                optimizer=self.governor_optimizer,
            )

            # Convert to symbolic movement command
            movement_logits = outputs["movement_logits"]
            movement_command_idx = int(torch.argmax(movement_logits, dim=-1).item())
            movement_command = self.movement_symbols[movement_command_idx]

            # Get decision confidence (max probability)
            movement_probs = torch.softmax(movement_logits, dim=-1)
            decision_confidence = float(torch.max(movement_probs).item())

            # MCTS maintenance
            decay_period = self.cfg.get("mcts", {}).get("decay_period", 6000)
            prune_period = self.cfg.get("mcts", {}).get("prune_period", 6000)

            if self.governor_counter % decay_period == 0:
                self.titans_alphasr.decay_visits()

            if self.governor_counter % prune_period == 0:
                self.titans_alphasr.prune_states()

            self.governor_counter += 1

            governor_info = {
                "movement_command": movement_command,
                "decision_confidence": decision_confidence,
                "reward": float(outputs.get("reward", 0.0)),
                "movement_probs": movement_probs.detach().cpu().numpy().tolist(),
                "counter": self.governor_counter,
            }

            return governor_info

        except Exception:
            return {"movement_command": "idle", "decision_confidence": 0.0}
