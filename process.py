import json
import time
import torch
import torch.nn as nn
import zmq
import numpy as np
from typing import Dict, Any, List
from torchvision import transforms
from importlib import import_module

from src.utilities.sensor.raycast import (
    process_line_of_sight,
    create_raycast_matrices,
)
from src.utilities.sensor.vision import create_lidar_vision_tensor
from agi.learning.perceive import Retina
from agi.memory.store import MemoryStore
from agi.learning.conscious.titans import Titans
from agi.learning.emotions import TitansAlphaSR
from pprint import pprint

# Vision preprocessing constants
vision_mean = [0.485, 0.456, 0.406]
vision_std = [0.229, 0.224, 0.225]


def vision_preproc(x) -> torch.Tensor | None:
    """Vision preprocessing function"""
    print(
        f"[DEBUG] vision_preproc input: type={type(x)}, shape={x.shape if hasattr(x, 'shape') else 'no shape'}"
    )

    if x is None:
        print("[DEBUG] vision_preproc: input is None")
        return None

    if isinstance(x, np.ndarray):
        print(
            f"[DEBUG] vision_preproc: converting numpy array, shape={x.shape}, dtype={x.dtype}"
        )
        x = torch.from_numpy(x)

    if x.dtype != torch.float32:
        print(f"[DEBUG] vision_preproc: converting dtype from {x.dtype} to float32")
        x = x.float()

    print(f"[DEBUG] vision_preproc: shape before unsqueeze check: {x.shape}")
    if len(x.shape) == 3:
        print("[DEBUG] vision_preproc: adding batch dimension")
        x = x.unsqueeze(0)

    print(f"[DEBUG] vision_preproc: shape before permute check: {x.shape}")
    if x.shape[1] != 3:
        print(f"[DEBUG] vision_preproc: permuting from {x.shape} to channels-first")
        x = x.permute(0, 3, 1, 2)

    print(f"[DEBUG] vision_preproc: final shape before normalization: {x.shape}")
    print(
        f"[DEBUG] vision_preproc: tensor stats - min={x.min()}, max={x.max()}, mean={x.mean()}"
    )

    try:
        # Normalize
        normalize = transforms.Normalize(mean=vision_mean, std=vision_std)
        x = normalize(x / 255.0)
        print("[DEBUG] vision_preproc: normalization successful")
        return x
    except Exception as e:
        print(f"[DEBUG] vision_preproc: normalization failed with error: {e}")
        return None


def vae_loss_function(reconstructed, original, mu, logvar) -> torch.Tensor:
    """VAE loss function"""
    reconstruction_loss = nn.functional.mse_loss(
        reconstructed, original, reduction="sum"
    )
    kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = reconstruction_loss + kl_divergence_loss
    return total_loss


class SequentialProcessor:
    """
    Sequential processor - ZMQ + ctx_parser + memory_manager + perceiver + motivator + governor
    """

    def __init__(self, runtime_engine):
        self.runtime_engine = runtime_engine
        self.cfg = runtime_engine.get_metadata("config")
        self.device = runtime_engine.get_metadata("device")

        # Get embedding dimension from config
        self.emb_dim = self.cfg["perceivers"]["vision"]["emb_dim"]

        # Initialize components
        self._init_zmq()
        self._init_perceiver()
        self._init_memory_manager()
        self._init_motivator()
        self._init_governor()

    def _init_zmq(self):
        """Initialize ZMQ connections"""
        self.context = zmq.Context()

        # Subscriber for receiving sensor data
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(
            f"tcp://{self.cfg['robot']['sub']['ip']}:{self.cfg['robot']['sub']['port']}"
        )
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")

        # Publisher for sending processed data
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.connect(
            f"tcp://{self.cfg['robot']['pub']['ip']}:{self.cfg['robot']['pub']['port']}"
        )

    def _init_perceiver(self):
        """Initialize the vision perceiver (Retina VAE)"""
        # Initialize Retina model
        self.retina = Retina(emb_dim=self.emb_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.retina.parameters(), lr=0.001)

    def _init_memory_manager(self):
        """Initialize memory management"""
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

    def _init_motivator(self):
        """Initialize motivator (intrinsics system)"""
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

    def perceive_vision(self, vision_data) -> torch.Tensor | None:
        """Process vision data through the perceiver"""
        print(
            f"[DEBUG] perceive_vision called with vision_data type: {type(vision_data)}"
        )

        if vision_data is None:
            print("[DEBUG] vision_data is None, returning early")
            return None

        # Preprocess vision data
        x = vision_preproc(vision_data)
        print(
            f"[DEBUG] vision_preproc returned: {type(x)}, shape: {x.shape if x is not None else 'None'}"
        )

        if x is None:
            print("[DEBUG] vision preprocessing returned None")
            return None

        x = x.to(self.device).unsqueeze(0)
        x0 = x.clone()

        print("[DEBUG] About to compute forward pass")

        # Forward pass
        reconstructed, mu, logvar = self.retina(x)

        # Normalize mu
        mu_normalized = mu / torch.linalg.norm(mu, ord=2, dim=1, keepdim=True)

        # Check for NaN values
        if torch.any(torch.isnan(mu_normalized)):
            print("[DEBUG] NaN detected in mu_normalized")
            return None

        print("[DEBUG] About to compute loss")

        # Backward pass
        loss = vae_loss_function(reconstructed, x0, mu_normalized, logvar)
        print("loss", loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return mu_normalized.detach()

    def motivator_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
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
                    except:
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

    def ctx_parser(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse context from sensor data"""
        try:
            # Extract data directly from raw_data (not nested under PerceptionData)
            line_of_sight = raw_data.get("line_of_sight", [])
            vision_data = raw_data.get("VisionData")  # This might not exist yet

            # Extract agent state information
            agent_position = {
                "x": raw_data.get("x", 0.0),
                "y": raw_data.get("y", 0.0),
                "z": raw_data.get("z", 0.0),
            }
            agent_state = raw_data.get("state", 0)  # 0=normal, 1=pain, 3=falling, etc.
            hit_point = raw_data.get("hit_point", 100)
            hunger = raw_data.get("hunger", 0.0)

            # Process raycast data
            raycast_info = process_line_of_sight(line_of_sight)

            # Create raycast matrices
            raycast_matrices = create_raycast_matrices(raycast_info)

            # Construct vision data from raycast if no vision data available
            if vision_data is None:
                vision_data = self._construct_vision_from_raycast(line_of_sight)

            # Process vision data through perceiver
            vision_latent = None
            if vision_data is not None:
                vision_latent = self.perceive_vision(vision_data)

            # Create lidar vision tensor
            lidar_vision_tensor = create_lidar_vision_tensor(raycast_matrices)

            # Categorize raycast detections by type
            detection_summary = self._categorize_detections(raycast_info)

            parsed_context = {
                "raycast_info": raycast_info,
                "raycast_matrices": raycast_matrices,
                "vision_latent": vision_latent,
                "lidar_vision_tensor": lidar_vision_tensor,
                "agent_position": agent_position,
                "agent_state": agent_state,
                "hit_point": hit_point,
                "hunger": hunger,
                "detection_summary": detection_summary,
                "raw_perception": raw_data,
            }

            return parsed_context

        except Exception:
            return {}

    def _categorize_detections(self, raycast_info: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize detections by type based on the type encoding"""
        type_mapping = {
            0: "nil",
            1: "obstacle",
            2: "enemy",
            3: "trap",
            4: "goal",
            5: "people",
            6: "food",
        }

        categorized = {
            name: {"count": 0, "distances": [], "angles": []}
            for name in type_mapping.values()
        }

        distances = raycast_info.get("distances", [])
        angles = raycast_info.get("angles", [])
        types = raycast_info.get("types", [])

        for i, (distance, angle, ray_type) in enumerate(zip(distances, angles, types)):
            type_name = type_mapping.get(ray_type, "unknown")
            if type_name != "nil":  # Skip nil detections
                categorized[type_name]["count"] += 1
                categorized[type_name]["distances"].append(distance)
                categorized[type_name]["angles"].append(angle)

        return categorized

    def memory_manager_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory operations"""
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

    def fifo_context_update(self, ctx, x):
        """FIFO context update for maintaining temporal context"""
        x = x.to(self.device)
        return torch.cat((ctx[:, 1:, :], x.unsqueeze(1)), dim=1)

    def governor_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
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

    def _tensor_to_serializable(self, obj):
        """Convert tensors and other non-serializable objects to serializable format"""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._tensor_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._tensor_to_serializable(item) for item in obj]
        else:
            return obj

    def process_step(self):
        """Process one step sequentially"""
        try:
            # 1. Receive sensor data via ZMQ
            if self.subscriber.poll(timeout=100):  # 100ms timeout
                raw_message = self.subscriber.recv_string(zmq.NOBLOCK)
                raw_data = json.loads(raw_message)

                # 2. Parse context
                context = self.ctx_parser(raw_data)
                if not context:
                    return

                # 3. Update memory
                memory_info = self.memory_manager_step(context)

                # 4. Generate emotion guidance through motivator
                motivator_info = self.motivator_step(context)

                # 5. Make decisions through governor
                governor_info = self.governor_step(
                    {
                        **context,
                        "emotion_guidance": motivator_info.get("emotion_guidance"),
                    }
                )

                # 6. Send processed data - Convert tensors to serializable format
                output = {
                    "timestamp": time.time(),
                    "memory_info": memory_info,
                    "motivator_info": self._tensor_to_serializable(motivator_info),
                    "governor_info": self._tensor_to_serializable(governor_info),
                    "has_vision_latent": context.get("vision_latent") is not None,
                    "raycast_count": len(
                        context.get("raycast_info", {}).get("distances", [])
                    ),
                    "emotion_guidance_available": motivator_info.get("emotion_guidance")
                    is not None,
                    "detection_summary": context.get("detection_summary", {}),
                    "agent_status": {
                        "state": context.get("agent_state", 0),
                        "hit_point": context.get("hit_point", 100),
                        "hunger": context.get("hunger", 0.0),
                        "position": context.get("agent_position", {}),
                    },
                    "action": {
                        "movement": governor_info.get("movement_command", "idle"),
                        "confidence": governor_info.get("decision_confidence", 0.0),
                    },
                }

                self.publisher.send_string(json.dumps(output))

        except zmq.Again:
            # No message available
            pass
        except Exception:
            pass

    def run(self):
        """Main processing loop"""
        try:
            while True:
                self.process_step()
                time.sleep(0.01)  # Small delay to prevent CPU spinning

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.subscriber.close()
        self.publisher.close()
        self.context.term()

    def _construct_vision_from_raycast(
        self, line_of_sight: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Construct a VAE-compatible vision matrix from raycast data"""
        # VAE-optimized constants
        max_distance = 100.0
        height = 64  # Standard VAE input size
        width = 64  # Standard VAE input size

        # Use 3 channels for RGB-like representation (compatible with vision_preproc)
        vision_matrix = np.zeros((height, width, 3), dtype=np.float32)

        if not line_of_sight:
            return vision_matrix

        num_rays = len(line_of_sight)

        # Create depth map, object type map, and intensity map
        for i, ray in enumerate(line_of_sight):
            distance = ray.get("Distance", max_distance)
            obj_type = ray.get("Type", 0)

            # Normalize distance (0-1 range, inverted so closer = brighter)
            depth_value = 1.0 - min(distance / max_distance, 1.0)

            # Map ray index to column
            col = int((i / max(num_rays - 1, 1)) * (width - 1))

            # Calculate row based on distance (closer objects appear lower)
            row = int((1.0 - depth_value) * (height - 1))

            # Channel 0: Depth/Distance (like red channel)
            vision_matrix[row, col, 0] = depth_value

            # Channel 1: Object type intensity (like green channel)
            # Map object types to different intensities
            type_intensity = min(obj_type / 6.0, 1.0) if obj_type > 0 else 0.0
            vision_matrix[row, col, 1] = type_intensity

            # Channel 2: Composite/Edge information (like blue channel)
            # Combine distance and type for richer representation
            vision_matrix[row, col, 2] = (depth_value + type_intensity) / 2.0

            # Create vertical spread for more realistic vision
            for r_offset in range(-3, 4):  # Increased spread
                new_row = row + r_offset
                if 0 <= new_row < height:
                    fade_factor = max(0, 1.0 - abs(r_offset) * 0.2)

                    # Apply faded values to all channels
                    for channel in range(3):
                        current_val = vision_matrix[new_row, col, channel]
                        new_val = vision_matrix[row, col, channel] * fade_factor
                        vision_matrix[new_row, col, channel] = max(current_val, new_val)

        # Enhanced smoothing for better VAE compatibility
        kernel_size = 3
        for channel in range(3):
            # Apply gaussian-like smoothing
            smoothed_channel = np.copy(vision_matrix[:, :, channel])
            for row in range(kernel_size // 2, height - kernel_size // 2):
                for col in range(kernel_size // 2, width - kernel_size // 2):
                    # Simple box filter
                    neighborhood = vision_matrix[
                        row - kernel_size // 2 : row + kernel_size // 2 + 1,
                        col - kernel_size // 2 : col + kernel_size // 2 + 1,
                        channel,
                    ]
                    if np.any(neighborhood > 0):
                        smoothed_channel[row, col] = (
                            np.mean(neighborhood[neighborhood > 0]) * 0.7
                        )

            vision_matrix[:, :, channel] = smoothed_channel

        # Scale to 0-255 range for compatibility with vision_preproc
        vision_matrix = (vision_matrix * 255.0).astype(np.uint8)

        return vision_matrix

    def _construct_polar_vision_from_raycast(
        self, line_of_sight: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Alternative: Construct polar coordinate vision from raycast data"""
        height = 64
        width = 64
        max_distance = 100.0

        # Initialize as RGB image
        vision_matrix = np.zeros((height, width, 3), dtype=np.float32)

        if not line_of_sight:
            return (vision_matrix * 255.0).astype(np.uint8)

        num_rays = len(line_of_sight)
        center_x, center_y = width // 2, height // 2

        for i, ray in enumerate(line_of_sight):
            distance = ray.get("Distance", max_distance)
            obj_type = ray.get("Type", 0)

            # Convert to polar coordinates
            angle = (i / max(num_rays - 1, 1)) * 2 * np.pi  # Full 360 degrees
            normalized_distance = min(distance / max_distance, 1.0)

            # Convert to cartesian coordinates for image
            radius = normalized_distance * min(center_x, center_y)
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))

            # Ensure coordinates are within bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))

            # Set pixel values
            intensity = 1.0 - normalized_distance  # Closer = brighter
            type_intensity = min(obj_type / 6.0, 1.0) if obj_type > 0 else 0.0

            vision_matrix[y, x, 0] = intensity
            vision_matrix[y, x, 1] = type_intensity
            vision_matrix[y, x, 2] = (intensity + type_intensity) / 2.0

            # Draw line from center to point for radar-like effect
            line_points = self._bresenham_line(center_x, center_y, x, y)
            for lx, ly in line_points:
                if 0 <= lx < width and 0 <= ly < height:
                    fade = max(0, intensity * 0.3)  # Faded line
                    vision_matrix[ly, lx, 0] = max(vision_matrix[ly, lx, 0], fade)

        return (vision_matrix * 255.0).astype(np.uint8)

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[tuple]:
        """Bresenham's line algorithm for drawing lines"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy

        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break

            e2 = 2 * error
            if e2 > -dy:
                error -= dy
                x += x_inc
            if e2 < dx:
                error += dx
                y += y_inc

        return points
