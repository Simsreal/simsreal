import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from pprint import pprint

from agi.learning.conscious.titans import Titans
from agi.learning.alphasr import TitansAlphaSR  # Updated import


class Governor:
    """Governor system for decision making and movement control"""

    def __init__(
        self,
        cfg: Dict[str, Any],
        device: torch.device,
        emb_dim: int,
        intrinsics: List[str],
        motivator=None,
    ):
        """
        Initialize governor system

        Args:
            cfg: Configuration dictionary containing brain settings
            device: PyTorch device for tensor operations
            emb_dim: Embedding dimension for latent representations
            intrinsics: List of intrinsic names for policy dimension calculation
            motivator: Motivator instance for emotion guidance
        """
        self.cfg = cfg
        self.device = device
        self.emb_dim = emb_dim
        self.intrinsics = intrinsics
        self.motivator = motivator

        # MCTS simulation parameters
        self.simulation_steps = cfg.get("mcts", {}).get("simulation_steps", 5)
        self.discount_factor = cfg.get("mcts", {}).get("discount_factor", 0.95)

        # Initialize governor components
        self._init_governor()

    def _init_governor(self):
        """Initialize the governor (decision making system)"""
        try:
            # Movement symbols - updated to match Unity AgentController
            self.movement_symbols = [
                "moveforward",
                "movebackward",
                "moveleft",      # Added - strafe left
                "moveright",     # Added - strafe right  
                "lookleft",
                "lookright",
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

            # Initialize TitansAlphaSR with mindmap integration
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

    def create_state_tensor_from_mindmap(self, context: Dict[str, Any]) -> torch.Tensor:
        """Create simplified state tensor from key simulator states"""
        try:
            raw_context = context.get("raw_context", {})
            
            # Core simulator states (directly from Unity)
            agent_state = raw_context.get("state", 0)        # 0=normal, 1=fell, 2=won, 3=dead
            agent_hunger = raw_context.get("hunger", 1.0)    # 0.0-1.0
            agent_orientation = raw_context.get("orientation", 0.0)  # 0-360 degrees
            
            # Simplified raycast representation (immediate surroundings)
            raycast_info = context.get("raycast_info", {})
            distances = raycast_info.get("distances", [])
            types = raycast_info.get("types", [])
            
            # Create symbolic state vector
            state_features = []
            
            # 1. Agent status (one-hot encoded for clarity)
            status_onehot = [0.0, 0.0, 0.0, 0.0]  # [normal, fell, won, dead]
            if 0 <= agent_state <= 3:
                status_onehot[agent_state] = 1.0
            state_features.extend(status_onehot)
            
            # 2. Agent hunger (normalized)
            state_features.append(agent_hunger)
            
            # 3. Agent orientation (normalized to [-1, 1] using sin/cos for continuity)
            orientation_rad = np.deg2rad(agent_orientation)
            state_features.extend([
                np.sin(orientation_rad),  # x-component
                np.cos(orientation_rad),  # y-component
            ])
            
            # 4. Immediate surroundings (simplified to 8 directions)
            # Group raycasts into 8 cardinal/diagonal directions
            directions = {
                'forward': [],      # 0° ± 22.5°
                'forward_right': [],  # 45° ± 22.5°
                'right': [],        # 90° ± 22.5°
                'back_right': [],   # 135° ± 22.5°
                'back': [],         # 180° ± 22.5°
                'back_left': [],    # 225° ± 22.5°
                'left': [],         # 270° ± 22.5°
                'forward_left': [], # 315° ± 22.5°
            }
            
            # Simple directional encoding (8 directions)
            for i, (distance, obj_type) in enumerate(zip(distances[:8], types[:8])):
                direction_idx = i % 8
                direction_names = list(directions.keys())
                if direction_idx < len(direction_names):
                    directions[direction_names[direction_idx]].append((distance, obj_type))
            
            # Encode each direction as [distance, has_obstacle, has_goal, has_trap, has_food]
            for direction in ['forward', 'forward_right', 'right', 'back_right', 
                            'back', 'back_left', 'left', 'forward_left']:
                if directions[direction]:
                    distance, obj_type = directions[direction][0]  # Take first hit
                    state_features.extend([
                        min(distance / 10.0, 1.0),  # Normalized distance (cap at 10 units)
                        1.0 if obj_type == 1 else 0.0,  # Has obstacle
                        1.0 if obj_type == 4 else 0.0,  # Has goal
                        1.0 if obj_type == 3 else 0.0,  # Has trap
                        1.0 if obj_type == 6 else 0.0,  # Has food
                    ])
                else:
                    # No hit in this direction
                    state_features.extend([1.0, 0.0, 0.0, 0.0, 0.0])  # Max distance, no objects
            
            # Convert to tensor
            state_tensor = torch.tensor(state_features, dtype=torch.float32, device=self.device)
            
            # Ensure it matches embedding dimension by padding or truncating
            if state_tensor.shape[0] > self.emb_dim:
                state_tensor = state_tensor[:self.emb_dim]
            elif state_tensor.shape[0] < self.emb_dim:
                padding = torch.zeros(self.emb_dim - state_tensor.shape[0], 
                                    dtype=torch.float32, device=self.device)
                state_tensor = torch.cat([state_tensor, padding], dim=0)
            
            return state_tensor
            
        except Exception as e:
            print(f"State tensor creation error: {e}")
            return torch.zeros(self.emb_dim, dtype=torch.float32, device=self.device)

    def simulate_action_sequence_with_intrinsics(self, context: Dict[str, Any], initial_action: str) -> float:
        """Simulate a sequence of actions with proper intrinsic reward accumulation"""
        try:
            if self.motivator is None or not self.motivator.motivators:
                return 0.0
            
            # Start simulation for all intrinsics
            for intrinsic in self.motivator.motivators.values():
                intrinsic.start_simulation(context)
            
            # Simulate action sequence
            current_context = context.copy()
            total_reward = 0.0
            
            for step in range(self.simulation_steps):
                # Use initial action for first step, then random exploration
                if step == 0:
                    action = initial_action
                else:
                    action = np.random.choice(self.movement_symbols)
                
                # Simulate state transition
                current_context = self._simulate_state_transition(current_context, action)
                
                # Get intrinsic rewards for this step
                step_reward = 0.0
                for intrinsic_name, intrinsic in self.motivator.motivators.items():
                    intrinsic_step_reward = intrinsic.simulate_step(current_context, action, step)
                    step_reward += intrinsic_step_reward
                
                # Apply discount factor
                discounted_reward = step_reward * (self.discount_factor ** step)
                total_reward += discounted_reward
                
                # Early termination if agent dies
                if current_context.get("raw_context", {}).get("state", 0) == 3:
                    break
            
            # End simulation for all intrinsics and get final accumulated rewards
            final_intrinsic_rewards = {}
            for intrinsic_name, intrinsic in self.motivator.motivators.items():
                final_reward = intrinsic.end_simulation()
                final_intrinsic_rewards[intrinsic_name] = final_reward
            
            # Add spatial analysis reward
            spatial_reward = self._analyze_spatial_context_for_action(context, initial_action)
            total_reward += spatial_reward
            
            return max(-1.0, min(1.0, total_reward))
            
        except Exception as e:
            print(f"Action sequence simulation error: {e}")
            return 0.0

    def _simulate_state_transition(self, context: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Simulate state transition after taking an action"""
        new_context = context.copy()
        raw_context = new_context.get("raw_context", {}).copy()
        
        # Simulate state changes based on action
        current_state = raw_context.get("state", 0)
        current_hunger = raw_context.get("hunger", 1.0)
        current_orientation = raw_context.get("orientation", 0.0)
        
        # State transitions
        if action == "standup" and current_state == 1:
            raw_context["state"] = 0  # Recover from fallen state
        elif action in ["moveforward", "movebackward", "moveleft", "moveright"]:
            # Movement actions consume hunger
            raw_context["hunger"] = max(0.0, current_hunger - 0.02)
            
            # Simulate potential hazards (simplified)
            if np.random.random() < 0.05:  # 5% chance of encountering hazard
                hazard_type = np.random.choice(["trap", "obstacle"])
                if hazard_type == "trap":
                    raw_context["state"] = 1  # Fall down
                # Note: We don't simulate death here to keep simulation optimistic
        
        elif action in ["lookleft", "lookright"]:
            # Turning actions change orientation
            turn_amount = 45 if action == "lookright" else -45
            raw_context["orientation"] = (current_orientation + turn_amount) % 360
        
        # Hunger death condition
        if raw_context["hunger"] <= 0.0:
            raw_context["state"] = 3  # Dead from hunger
        
        new_context["raw_context"] = raw_context
        return new_context

    def _analyze_spatial_context_for_action(self, context: Dict[str, Any], action: str) -> float:
        """Analyze spatial context from local mindmap for action evaluation"""
        try:
            local_mindmap = context.get("local_mind_map")
            if local_mindmap is None:
                return 0.0
            
            raw_context = context.get("raw_context", {})
            orientation = raw_context.get("orientation", 0.0)
            
            # local_mindmap is [7, 128, 128] - [obstacle, checkpoint, trap, goal, people, food, explored]
            map_data = local_mindmap.cpu().numpy()
            center = 64  # Center of 128x128 map
            
            # Define action directions relative to agent orientation
            action_directions = {
                "moveforward": 0,
                "movebackward": 180,
                "moveleft": 270,
                "moveright": 90,
                "lookleft": orientation - 45,
                "lookright": orientation + 45,
                "standup": None,
            }
            
            if action not in action_directions or action_directions[action] is None:
                return 0.0
            
            # Calculate target direction
            target_direction = (orientation + action_directions[action]) % 360
            direction_rad = np.deg2rad(target_direction)
            
            # Check areas in the direction of movement
            spatial_reward = 0.0
            
            # Sample points in the direction of action
            for distance in [5, 10, 15, 20]:
                dx = int(distance * np.sin(direction_rad))
                dy = int(distance * np.cos(direction_rad))
                
                check_x = center + dx
                check_y = center + dy
                
                if 0 <= check_x < 128 and 0 <= check_y < 128:
                    # Reward for moving toward goals and food
                    if map_data[3, check_y, check_x] > 0.1:  # Goal
                        spatial_reward += 0.5 / distance
                    if map_data[5, check_y, check_x] > 0.1:  # Food
                        spatial_reward += 0.3 / distance
                    
                    # Penalty for moving toward obstacles and traps
                    if map_data[0, check_y, check_x] > 0.1:  # Obstacle
                        spatial_reward -= 0.4 / distance
                    if map_data[2, check_y, check_x] > 0.1:  # Trap
                        spatial_reward -= 0.6 / distance
                    
                    # Small reward for exploring unexplored areas
                    if map_data[6, check_y, check_x] < 0.1:  # Unexplored
                        spatial_reward += 0.1 / distance
            
            return spatial_reward
            
        except Exception as e:
            print(f"Spatial analysis error: {e}")
            return 0.0

    def fifo_context_update(self, ctx, x):
        """FIFO context update for maintaining temporal context"""
        x = x.to(self.device)
        
        # Ensure x has the right shape for concatenation
        if x.dim() == 1:
            # x is [emb_dim] -> reshape to [1, 1, emb_dim]
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            # x is [batch, emb_dim] -> reshape to [batch, 1, emb_dim]
            x = x.unsqueeze(1)
        
        # ctx is [batch, ctx_len, emb_dim]
        # x is now [batch, 1, emb_dim]
        return torch.cat((ctx[:, 1:, :], x), dim=1)

    def process_step(self, context: Dict[str, Any], skip_heavy_processing: bool = False) -> Dict[str, Any]:
        """Generate movement decisions using the governor with accumulated intrinsic rewards"""
        try:
            if self.titans_alphasr is None:
                return {"movement_command": "idle", "decision_confidence": 0.0}

            # Get emotion guidance from motivator
            emotion_guidance = None
            motivator_info = {}
            
            if not skip_heavy_processing and self.motivator is not None:
                motivator_info = self.motivator.process_step(context)
                emotion_guidance = motivator_info.get("emotion_guidance")
            else:
                if self.motivator is not None:
                    emotion_guidance = getattr(self.motivator, 'current_emotion_guidance', None)

            # Create rich state tensor from mindmap
            state_tensor = self.create_state_tensor_from_mindmap(context)

            # Update context with FIFO
            self.ctx = self.fifo_context_update(self.ctx, state_tensor)

            # Enhanced simulation with proper intrinsic reward accumulation
            action_rewards = {}
            intrinsic_breakdowns = {}
            
            for action in self.movement_symbols:
                # Simulate action sequence with accumulated intrinsic rewards
                accumulated_reward = self.simulate_action_sequence_with_intrinsics(context, action)
                action_rewards[action] = accumulated_reward
                
                # Store breakdown for debugging
                intrinsic_breakdowns[action] = {
                    "total_reward": accumulated_reward,
                    "simulation_steps": self.simulation_steps,
                }

            # Use best simulated reward for MCTS
            best_reward = max(action_rewards.values())

            # Get outputs from TitansAlphaSR with enhanced simulation
            outputs = self.titans_alphasr.forward(
                state_tensor,
                self.ctx,
                emotion_guidance=emotion_guidance,
                optimizer=self.governor_optimizer,
                simulated_reward=best_reward,
                action_rewards=action_rewards,
            )

            # Convert to symbolic movement command
            movement_logits = outputs["movement_logits"]
            movement_command_idx = int(torch.argmax(movement_logits, dim=-1).item())
            movement_command = self.movement_symbols[movement_command_idx]

            # Get decision confidence
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
                "simulated_reward": best_reward,
                "action_rewards": action_rewards,
                "intrinsic_breakdowns": intrinsic_breakdowns,
                "movement_probs": movement_probs.detach().cpu().numpy().tolist(),
                "counter": self.governor_counter,
                **motivator_info
            }

            return governor_info

        except Exception as e:
            print(f"Governor error: {e}")
            return {"movement_command": "idle", "decision_confidence": 0.0}
