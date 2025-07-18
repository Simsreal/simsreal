"""
Monte Carlo Tree Search implementation with policy-value network and intrinsic motivations.

Integrates with discrete state representation and intrinsic reward system for
efficient simulation-based planning in the SimsReal environment.
"""

import math
import random
import time
import copy
import threading
import uuid
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from loguru import logger

from simsreal_types import (
    State, DiscreteState, ActionType, 
    OptimizedDiscreteState,  # Add this
    state_to_discrete_state, discrete_state_to_hash,
    state_to_optimized_state, optimized_discrete_state_to_hash,  # Add these
    get_valid_actions, get_contextual_actions,
    apply_action_to_state, calculate_state_space_size,
    check_goal_reached, check_stagnation, simulate_hunger_decay
)
from intrinsics import IntrinsicSystem, create_intrinsic_system


class MCTSError(Exception):
    """Custom exception for MCTS failures"""
    pass


class PolicyValueNetwork(nn.Module):
    """
    Neural network that estimates policy probabilities and state values
    for MCTS guidance using discrete state representation.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256, num_actions: int = 5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size // 2, num_actions),
            nn.Softmax(dim=-1)
        )
        
        # Value head (state value estimation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network
        
        Args:
            state_tensor: Encoded discrete state tensor
            
        Returns:
            (policy_probs, state_value) tuple
        """
        shared_features = self.shared_layers(state_tensor)
        policy_probs = self.policy_head(shared_features)
        state_value = self.value_head(shared_features)
        
        return policy_probs, state_value


class RolloutSummary:
    """Tracks rollout statistics for summarized logging"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.rollout_count = 0
        self.total_reward = 0.0
        self.rewards = []
        self.depths = []
        self.durations = []
        self.termination_reasons = defaultdict(int)
        self.intrinsic_details = defaultdict(float)
    
    def add_rollout(self, reward: float, depth: int, duration: float, 
                   termination_reason: str = "max_depth",
                   intrinsic_breakdown: Dict[str, float] = None):
        self.rollout_count += 1
        self.total_reward += reward
        self.rewards.append(reward)
        self.depths.append(depth)
        self.durations.append(duration)
        self.termination_reasons[termination_reason] += 1
        
        if intrinsic_breakdown:
            for name, value in intrinsic_breakdown.items():
                self.intrinsic_details[name] += value
    
    def get_summary(self) -> Dict[str, Any]:
        if self.rollout_count == 0:
            return {"rollouts": 0}
        
        return {
            "rollouts": self.rollout_count,
            "avg_reward": self.total_reward / self.rollout_count,
            "total_reward": self.total_reward,
            "reward_range": (min(self.rewards), max(self.rewards)) if self.rewards else (0, 0),
            "avg_depth": sum(self.depths) / len(self.depths) if self.depths else 0,
            "avg_duration": sum(self.durations) / len(self.durations) if self.durations else 0,
            "termination_reasons": dict(self.termination_reasons),
            "intrinsic_totals": dict(self.intrinsic_details)
        }


class RolloutContext:
    """
    Context manager for MCTS rollout that ensures fresh intrinsic system
    and tracks exploration data
    """
    def __init__(self, mcts_instance, rollout_id: str = None, summary: RolloutSummary = None,
                 snapshot_manager=None, mcts_iteration: int = 0, root_state: State = None):
        self.mcts = mcts_instance
        self.rollout_id = rollout_id or str(uuid.uuid4())[:8]
        self.summary = summary
        self.snapshot_manager = snapshot_manager
        self.mcts_iteration = mcts_iteration
        self.root_state = root_state
        self.intrinsic_system = None
        self.created_at = time.time()
        self.exploration_id = None
        self.step_count = 0
        self.total_reward = 0.0
        self.termination_reason = "max_depth"
        
    def __enter__(self) -> IntrinsicSystem:
        """Create fresh intrinsic system for this rollout and start exploration tracking"""
        try:
            self.intrinsic_system = self.mcts.create_fresh_intrinsic_system(self.rollout_id)
            
            # Link the intrinsic system back to this context for tracking
            self.intrinsic_system._rollout_context = self
            
            # Start exploration tracking if snapshot manager is available
            if self.snapshot_manager and self.root_state:
                self.exploration_id = self.snapshot_manager.create_exploration_session(
                    self.mcts_iteration, self.root_state
                )
            
            return self.intrinsic_system
        except Exception as e:
            logger.error(f"Rollout {self.rollout_id}: Failed to create context: {e}")
            raise MCTSError(f"Rollout context creation failed: {e}")
    
    def track_step(self, step: int, state: State, action: ActionType, 
                   reward: float, is_terminal: bool, intrinsic_breakdown: Dict[str, float] = None):
        """Track a step in the rollout for exploration analysis"""
        self.step_count = step + 1
        self.total_reward += reward
        
        if is_terminal:
            if check_goal_reached(state):
                self.termination_reason = "goal_reached"
            elif state['hunger'] <= 0:
                self.termination_reason = "hunger"
            elif check_stagnation(state["location"], state["location"]):  # This would need previous location
                self.termination_reason = "stagnation"
            else:
                self.termination_reason = "terminal_state"
        
        # Save exploration step if tracking enabled
        if self.snapshot_manager and self.exploration_id:
            additional_info = {
                "intrinsic_rewards": intrinsic_breakdown or {},
                "rollout_id": self.rollout_id,
                "mcts_iteration": self.mcts_iteration
            }
            
            self.snapshot_manager.save_exploration_step(
                self.exploration_id, step, state, action, reward, is_terminal, additional_info
            )
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up rollout context and update summary"""
        duration = time.time() - self.created_at
        
        if self.intrinsic_system and self.summary:
            # Get intrinsic breakdown for summary
            intrinsic_info = self.intrinsic_system.get_intrinsic_info()
            intrinsic_breakdown = {}

            # Get system-level total reward
            system_total = intrinsic_info.get("total_reward_calculated", 0.0)

            # Calculate breakdown proportionally or track per-intrinsic totals
            for name, info in intrinsic_info["intrinsics"].items():
                # Use last_reward as it's the available data
                intrinsic_breakdown[name] = info.get("last_reward", 0.0)
            
            # Add to summary
            self.summary.add_rollout(
                self.total_reward, self.step_count, duration, 
                self.termination_reason, intrinsic_breakdown
            )
        
        # Finalize exploration tracking
        if self.snapshot_manager and self.exploration_id:
            rollout_summary = {
                "rollout_id": self.rollout_id,
                "mcts_iteration": self.mcts_iteration,
                "intrinsic_breakdown": intrinsic_breakdown if self.intrinsic_system else {}
            }
            
            self.snapshot_manager.finalize_exploration_session(
                self.exploration_id, self.total_reward, self.step_count,
                self.termination_reason, rollout_summary
            )
        
        # Clear intrinsic system
        self.intrinsic_system = None


class MCTSNode:
    """MCTS tree node representing a game state"""
    
    def __init__(self, state: State, parent: Optional['MCTSNode'] = None, 
                 action_taken: Optional[ActionType] = None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children: Dict[ActionType, 'MCTSNode'] = {}
        
        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0
        
        # Valid actions for this state
        self.valid_actions = get_contextual_actions(state['state'])
        self.is_terminal = len(self.valid_actions) == 0 or state['state'] in [2, 3]  # won or dead
        
        # State encoding for neural network
        self.discrete_state = state_to_discrete_state(state)
        self.state_hash = discrete_state_to_hash(self.discrete_state)
    
    def is_fully_expanded(self) -> bool:
        """Check if all valid actions have been expanded"""
        return len(self.children) == len(self.valid_actions)
    
    def get_unexplored_actions(self) -> List[ActionType]:
        """Get list of actions that haven't been expanded yet"""
        return [action for action in self.valid_actions if action not in self.children]
    
    def add_child(self, action: ActionType, child_state: State) -> 'MCTSNode':
        """Add a child node for the given action"""
        child = MCTSNode(child_state, parent=self, action_taken=action)
        self.children[action] = child
        return child
    
    def update(self, value: float):
        """Update node statistics with a new value"""
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count
    
    def ucb_score(self, exploration_constant: float, total_parent_visits: int) -> float:
        """Calculate UCB1 score for action selection"""
        if self.visit_count == 0:
            return float('inf')  # Prioritize unvisited nodes
        
        exploitation = self.mean_value
        exploration = exploration_constant * math.sqrt(
            math.log(total_parent_visits) / self.visit_count
        )
        return exploitation + exploration


class MCTS:
    """
    Monte Carlo Tree Search implementation with policy-value network guidance
    and intrinsic motivation rewards. Each rollout uses fresh intrinsic state.
    GPU acceleration is required and enforced. Integrates with exploration tracking.
    """
    
    def __init__(self, config: Dict[str, Any], snapshot_manager=None):
        self.config = config
        self.mcts_config = config.get('mcts', {})
        self.snapshot_manager = snapshot_manager
        
        # MCTS parameters
        self.max_iterations = self.mcts_config.get('max_iterations', 500)
        self.max_depth = self.mcts_config.get('max_depth', 30)
        self.exploration_constant = self.mcts_config.get('exploration_constant', 1.414)
        self.simulation_steps = self.mcts_config.get('simulation_steps', 20)
        self.max_retries = self.mcts_config.get('max_retries', 3)
        self.timeout_seconds = self.mcts_config.get('timeout_seconds', 10.0)
        
        # Early termination settings
        self.enable_early_termination = self.mcts_config.get('enable_early_termination', True)
        self.restrict_to_explorable = self.mcts_config.get('restrict_to_explorable', True)
        self.simulation_time_step = self.mcts_config.get('simulation_time_step', 1.0)  # seconds per step
        
        # Exploration tracking settings
        self.enable_exploration_tracking = self.mcts_config.get('enable_exploration_tracking', True)
        
        # GPU enforcement
        self.require_gpu = self.mcts_config.get('require_gpu', True)
        self.use_gpu = self.mcts_config.get('use_gpu', True)
        
        # Initialize device with strict GPU checking
        self._initialize_device()
        
        # Base intrinsic system template (not used for rollouts)
        self.base_intrinsic_system = create_intrinsic_system()
        self._configure_intrinsics()
        
        # Initialize policy-value network (REQUIRED)
        self.policy_value_net = None
        self.optimizer = None
        self._initialize_network()
        
        # Add learning tracking
        self.learning_enabled = self.mcts_config.get('enable_learning', True)
        self.train_every_n_searches = self.mcts_config.get('train_every_n_searches', 10)
        self.batch_size = self.mcts_config.get('batch_size', 32)
        self.searches_since_training = 0
        
        # Training data collection
        self.training_data = {
            'states': [],
            'action_probs': [],
            'values': []
        }
        
        # Learning metrics for validation
        self.learning_metrics = {
            'policy_loss_history': [],
            'value_loss_history': [],
            'total_loss_history': [],
            'weight_changes': [],
            'gradient_norms': [],
            'training_count': 0,
            'last_training_time': None
        }
        
        # Store initial weights for comparison
        if self.policy_value_net:
            self.initial_weights = {
                name: param.clone().detach() 
                for name, param in self.policy_value_net.named_parameters()
            }
        
        # Statistics
        self.search_stats = {
            'iterations_performed': 0,
            'average_depth_reached': 0.0,
            'nodes_created': 0,
            'cache_hits': 0,
            'rollouts_performed': 0,
            'fresh_intrinsics_created': 0,
            'search_failures': 0,
            'timeouts': 0,
            'rollout_contexts_created': 0,
            'early_terminations': 0,
            'stagnation_terminations': 0,
            'hunger_terminations': 0,
            'goal_terminations': 0,
            'explorations_tracked': 0
        }
        
        # Node cache for repeated states
        self.node_cache: Dict[str, MCTSNode] = {}
        
        # Rollout summary for concise logging
        self.current_search_summary = RolloutSummary()
        
        logger.info(f"MCTS initialized with {self.max_iterations} iterations, "
                   f"exploration_constant={self.exploration_constant}")
        logger.info(f"Device: {self.device} (GPU required: {self.require_gpu})")
        logger.info(f"Early termination: {'ENABLED' if self.enable_early_termination else 'DISABLED'}")
        logger.info(f"Explorable area restriction: {'ENABLED' if self.restrict_to_explorable else 'DISABLED'}")
        logger.info(f"Exploration tracking: {'ENABLED' if self.enable_exploration_tracking and self.snapshot_manager else 'DISABLED'}")
        logger.info("Each MCTS rollout uses fresh intrinsic system - summaries will be logged")
        
        # Final validation
        self._validate_initialization()
    
    def _initialize_device(self):
        """Initialize compute device with GPU enforcement"""
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                error_msg = "GPU acceleration required but CUDA not available!"
                if self.require_gpu:
                    logger.error(error_msg)
                    raise MCTSError(error_msg)
                else:
                    logger.warning(f"{error_msg} Falling back to CPU.")
                    self.device = torch.device('cpu')
        else:
            if self.require_gpu:
                raise MCTSError("GPU is required but use_gpu is set to False in config!")
            self.device = torch.device('cpu')
        
        # Verify GPU memory if using GPU
        if self.device.type == 'cuda':
            try:
                # Test GPU memory allocation
                test_tensor = torch.randn(1000, 1000, device=self.device)
                del test_tensor
                torch.cuda.empty_cache()
                logger.info("GPU memory test passed")
            except Exception as e:
                error_msg = f"GPU memory test failed: {e}"
                logger.error(error_msg)
                if self.require_gpu:
                    raise MCTSError(error_msg)
    
    def _configure_intrinsics(self):
        """Configure intrinsic motivation weights from config"""
        intrinsic_config = self.mcts_config.get('intrinsics', {})
        
        for intrinsic_name, intrinsic in self.base_intrinsic_system.intrinsics.items():
            if intrinsic_name in intrinsic_config:
                intrinsic_settings = intrinsic_config[intrinsic_name]
                intrinsic.weight = intrinsic_settings.get('weight', intrinsic.weight)
                intrinsic.enabled = intrinsic_settings.get('enabled', intrinsic.enabled)
        
        logger.info("Base intrinsic system configured from config")
    
    def create_fresh_intrinsic_system(self, rollout_id: str = None) -> IntrinsicSystem:
        """
        Create a completely fresh intrinsic system for each rollout.
        Each rollout must have its own independent intrinsic state.
        
        Args:
            rollout_id: Optional identifier for tracking this rollout
            
        Returns:
            Fresh IntrinsicSystem instance
        """
        try:
            # Create completely new intrinsic system - no shared state
            fresh_system = create_intrinsic_system()
            
            # Copy configuration from base system (weights and enabled flags only)
            for name, base_intrinsic in self.base_intrinsic_system.intrinsics.items():
                if name in fresh_system.intrinsics:
                    fresh_intrinsic = fresh_system.intrinsics[name]
                    # Only copy configuration, not internal state
                    fresh_intrinsic.weight = base_intrinsic.weight
                    fresh_intrinsic.enabled = base_intrinsic.enabled
            
            # Copy system-level settings but ensure fresh state
            fresh_system.enabled = self.base_intrinsic_system.enabled
            fresh_system.debug_mode = False  # Always disable debug for rollouts
            
            # Track creation for statistics only
            self.search_stats['fresh_intrinsics_created'] += 1
            
            # Verify the system is properly isolated
            assert fresh_system is not self.base_intrinsic_system, "System not isolated!"
            assert id(fresh_system) != id(self.base_intrinsic_system), "Same instance returned!"
            
            return fresh_system
            
        except Exception as e:
            error_msg = f"Failed to create fresh intrinsic system for rollout {rollout_id}: {e}"
            logger.error(error_msg)
            raise MCTSError(error_msg)
    
    def get_rollout_context(self, rollout_id: str = None, mcts_iteration: int = 0, 
                           root_state: State = None) -> RolloutContext:
        """Get a new rollout context with fresh intrinsic system and exploration tracking"""
        self.search_stats['rollout_contexts_created'] += 1
        
        # Only enable exploration tracking if both enabled and snapshot manager available
        snapshot_manager = None
        if self.enable_exploration_tracking and self.snapshot_manager:
            snapshot_manager = self.snapshot_manager
            self.search_stats['explorations_tracked'] += 1
        
        return RolloutContext(self, rollout_id, self.current_search_summary, 
                            snapshot_manager, mcts_iteration, root_state)
    
    def _initialize_network(self):
        """Initialize policy-value network with configurable input size"""
        try:
            net_config = self.mcts_config.get('policy_value_net', {})
            hidden_size = net_config.get('hidden_size', 256)
            learning_rate = net_config.get('learning_rate', 0.001)
            
            # Determine input size based on state representation
            use_optimized = net_config.get('use_optimized_state', True)
            
            if use_optimized:
                input_size = 20  # Optimized: 1+1+2+16 = 20 features
                logger.info("Using optimized state representation (20 features)")
            else:
                input_size = 66  # Legacy: 1+1+(8×8) = 66 features
                logger.info("Using legacy state representation (66 features)")
            
            self.use_optimized_state = use_optimized
            
            self.policy_value_net = PolicyValueNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                num_actions=len(ActionType)
            ).to(self.device)
            
            self.optimizer = optim.Adam(
                self.policy_value_net.parameters(),
                lr=learning_rate
            )
            
            # Test network forward pass with correct input size
            test_input = torch.randn(1, input_size, device=self.device)
            with torch.no_grad():
                policy_out, value_out = self.policy_value_net(test_input)
            
            assert policy_out.shape == (1, len(ActionType)), f"Policy output shape mismatch: {policy_out.shape}"
            assert value_out.shape == (1, 1), f"Value output shape mismatch: {value_out.shape}"
            
            logger.info(f"Policy-value network initialized successfully")
            logger.info(f"Input size: {input_size}, Hidden size: {hidden_size}, Device: {self.device}")
            
        except Exception as e:
            error_msg = f"Failed to initialize policy-value network: {e}"
            logger.error(error_msg)
            raise MCTSError(error_msg)
    
    def _validate_initialization(self):
        """Final validation of MCTS initialization"""
        try:
            assert self.policy_value_net is not None, "Policy-value network not initialized"
            assert self.optimizer is not None, "Optimizer not initialized"
            assert self.device.type == 'cuda' or not self.require_gpu, "GPU required but not available"
            
            # Test fresh intrinsic system creation
            test_intrinsic = self.create_fresh_intrinsic_system("validation_test")
            assert test_intrinsic is not None, "Failed to create test intrinsic system"
            assert test_intrinsic is not self.base_intrinsic_system, "Intrinsic system not isolated"
            
            logger.info("MCTS initialization validation passed")
            
        except Exception as e:
            error_msg = f"MCTS initialization validation failed: {e}"
            logger.error(error_msg)
            raise MCTSError(error_msg)
    
    def encode_optimized_discrete_state(self, optimized_state: OptimizedDiscreteState) -> torch.Tensor:
        """
        Encode optimized discrete state into tensor for neural network input
        
        Input size: exactly 20 features
        - Agent status: 1 feature
        - Hunger level: 1 feature
        - Spatial context: 2 features (grid_x, grid_z only)
        - Zones: 16 features (4 zones × 4 features each)
        
        Args:
            optimized_state: Optimized discrete state representation
            
        Returns:
            Encoded state tensor (20 features)
        """
        try:
            features = []
            
            # Agent status (1 feature)
            features.append(float(optimized_state['agent_status']))
            
            # Hunger level (1 feature)
            features.append(float(optimized_state['hunger_level']) / 2.0)  # Normalize 0-2 to 0-1
            
            # Spatial context (2 features) - only grid position
            spatial = optimized_state['spatial_context']
            features.extend([
                float(spatial['grid_x']) / 7.0,  # Normalize to 0-1
                float(spatial['grid_z']) / 7.0,  # Normalize to 0-1
            ])
            
            # Zone information: exactly 4 zones × 4 features each = 16 features
            zones = optimized_state['zones']
            
            for zone_idx in range(4):
                if zone_idx < len(zones):
                    zone = zones[zone_idx]
                    zone_features = [
                        float(zone['zone']) / 3.0,              # Zone index 0-3 normalized
                        float(zone['has_obstacle']),            # Boolean 0/1
                        float(zone['important_object']) / 4.0,  # Normalize 0-4 to 0-1
                        float(zone['closest_distance']) / 2.0   # Normalize 0-2 to 0-1
                    ]
                else:
                    # Pad with empty zone
                    zone_features = [float(zone_idx) / 3.0, 0.0, 0.0, 0.0]
                
                features.extend(zone_features)
            
            tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            
            # Verify exactly 20 features: 1 + 1 + 2 + 16 = 20
            assert tensor.shape[0] == 20, f"Feature vector size mismatch: {tensor.shape[0]} != 20"
            
            return tensor
            
        except Exception as e:
            error_msg = f"Failed to encode optimized discrete state: {e}"
            logger.error(error_msg)
            raise MCTSError(error_msg)
    
    def get_network_prediction(self, state: State) -> Tuple[Dict[ActionType, float], float]:
        """
        Get policy probabilities and value estimation from neural network
        Now supports both optimized and legacy state representations
        """
        try:
            assert self.policy_value_net is not None, "Policy network not initialized"
            
            if getattr(self, 'use_optimized_state', True):  # Default to optimized
                # Use optimized state representation
                optimized_state = state_to_optimized_state(state)
                state_tensor = self.encode_optimized_discrete_state(optimized_state).unsqueeze(0)
            else:
                # Use legacy state representation
                discrete_state = state_to_discrete_state(state)
                state_tensor = self.encode_discrete_state(discrete_state).unsqueeze(0)
            
            with torch.no_grad():
                policy_probs, state_value = self.policy_value_net(state_tensor)
            
            # Convert to action probabilities dictionary
            valid_actions = get_contextual_actions(state['state'])
            action_probs = {}
            
            for i, action in enumerate(ActionType):
                if action in valid_actions:
                    prob = policy_probs[0, i].item()
                    assert 0.0 <= prob <= 1.0, f"Invalid probability: {prob}"
                    action_probs[action] = prob
            
            # Normalize probabilities for valid actions only
            total_prob = sum(action_probs.values())
            if total_prob > 0:
                action_probs = {action: prob / total_prob for action, prob in action_probs.items()}
            else:
                # Fallback to uniform distribution
                uniform_prob = 1.0 / len(valid_actions) if valid_actions else 0.0
                action_probs = {action: uniform_prob for action in valid_actions}
            
            value = state_value[0, 0].item()
            
            return action_probs, value
            
        except Exception as e:
            error_msg = f"Failed to get network prediction: {e}"
            logger.error(error_msg)
            raise MCTSError(error_msg)
    
    def simulate_with_fresh_intrinsics(self, state: State, rollout_id: str, 
                                     mcts_iteration: int = 0, root_state: State = None, 
                                     depth: int = 0) -> float:
        """
        Perform rollout simulation with guaranteed fresh intrinsic system and exploration tracking
        
        Args:
            state: Starting state for simulation
            rollout_id: Unique identifier for this rollout
            mcts_iteration: Current MCTS iteration number
            root_state: Original root state for exploration tracking
            depth: Current simulation depth
            
        Returns:
            Accumulated reward from simulation
        """
        try:
            # Use context manager to ensure fresh intrinsics and exploration tracking
            with self.get_rollout_context(rollout_id, mcts_iteration, root_state) as intrinsic_system:
                # Pass the initial hunger from the real state
                return self._simulate_rollout(state, intrinsic_system, depth, rollout_id, 
                                            state['hunger'], mcts_iteration)
                
        except Exception as e:
            error_msg = f"Rollout {rollout_id} failed at depth {depth}: {e}"
            logger.error(error_msg)
            raise MCTSError(error_msg)
    
    def _simulate_rollout(self, state: State, intrinsic_system: IntrinsicSystem, 
                         depth: int, rollout_id: str, initial_hunger: float = None,
                         mcts_iteration: int = 0) -> float:
        """
        Internal rollout simulation with provided intrinsic system and early termination
        """
        context = intrinsic_system._rollout_context if hasattr(intrinsic_system, '_rollout_context') else None
        
        if depth >= self.simulation_steps or state['state'] in [2, 3]:  # Terminal or max depth
            if context:
                context.track_step(depth, state, ActionType.MOVE_FORWARD, 0.0, True)
            return 0.0
        
        # Use initial hunger from real state if provided, otherwise use current state hunger
        if initial_hunger is None:
            initial_hunger = state['hunger']
        
        # Check early termination conditions if enabled
        if self.enable_early_termination:
            # Calculate hunger after time decay
            elapsed_time = depth * self.simulation_time_step
            current_hunger = simulate_hunger_decay(initial_hunger, elapsed_time)
            
            # Update state with decayed hunger
            state = state.copy()
            state['hunger'] = current_hunger
            
            # Check hunger termination
            if current_hunger <= 0:
                self.search_stats['hunger_terminations'] += 1
                self.search_stats['early_terminations'] += 1
                return -10.0  # Heavy penalty for starving
            
            # Check goal reached
            if check_goal_reached(state):
                self.search_stats['goal_terminations'] += 1
                self.search_stats['early_terminations'] += 1
                return 10.0  # High reward for reaching goal
        
        # Get valid actions
        valid_actions = get_contextual_actions(state['state'])
        if not valid_actions:
            return 0.0
        
        # Select a random action for faster rollouts
        action = random.choice(valid_actions)
        
        # Enforce environment rule: if fallen, must stand up. This is an impossible action.
        if state["state"] == 1 and action != ActionType.STANDUP:
            logger.warning(f"Rollout {rollout_id}: Invalid action {action.name} for fallen state. Terminating.")
            if context:
                context.track_step(depth, state, action, -10.0, True, 
                                   intrinsic_breakdown={"invalid_action": -10.0})
                context.termination_reason = "invalid_action_when_fallen"
            self.search_stats['early_terminations'] += 1
            return -10.0 # Heavy penalty for impossible action
        
        # Apply action with early termination check
        if self.enable_early_termination:
            next_state, should_terminate = apply_action_to_state(
                state, action, 
                restrict_to_explorable=self.restrict_to_explorable,
                simulate_time=False,  # We handle time separately
                time_step_seconds=self.simulation_time_step
            )
            
            # Check for stagnation
            if should_terminate and check_stagnation(state["location"], next_state["location"]):
                self.search_stats['stagnation_terminations'] += 1
                self.search_stats['early_terminations'] += 1
                return -5.0  # Penalty for stagnation
        else:
            # Use legacy function without termination
            from simsreal_types import apply_action_to_state_legacy
            next_state = apply_action_to_state_legacy(state, action)
            should_terminate = False
        
        # Calculate intrinsic reward using the provided fresh system
        reward = intrinsic_system.calculate_total_reward(state, action, next_state)
        
        # Track this step if exploration tracking is enabled
        if context:
            # Get intrinsic breakdown for tracking
            intrinsic_info = intrinsic_system.get_intrinsic_info()
            intrinsic_breakdown = {}
            for name, info in intrinsic_info["intrinsics"].items():
                intrinsic_breakdown[name] = info.get("last_reward", 0.0)
            
            context.track_step(depth, next_state, action, reward, should_terminate, intrinsic_breakdown)
        
        # Continue simulation if not terminated
        if should_terminate:
            return reward  # End simulation here
        else:
            future_reward = self._simulate_rollout(next_state, intrinsic_system, depth + 1, 
                                                 rollout_id, initial_hunger, mcts_iteration)
            return reward + 0.95 * future_reward  # Discount factor
    
    def select_child(self, node: MCTSNode) -> MCTSNode:
        """Select best child using UCB1 formula"""
        try:
            best_child = None
            best_score = float('-inf')
            
            for child in node.children.values():
                score = child.ucb_score(self.exploration_constant, node.visit_count)
                if score > best_score:
                    best_score = score
                    best_child = child
            
            assert best_child is not None, "No child selected"
            return best_child
            
        except Exception as e:
            error_msg = f"Child selection failed: {e}"
            logger.error(error_msg)
            raise MCTSError(error_msg)
    
    def expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Expand node by adding a new child"""
        try:
            if node.is_terminal or node.is_fully_expanded():
                return None
            
            unexplored_actions = node.get_unexplored_actions()
            if not unexplored_actions:
                return None
            
            # Select action to expand
            action = random.choice(unexplored_actions)
            
            # Apply action to get child state
            child_state, _ = apply_action_to_state(node.state, action)
            
            # Create and add child node
            child = node.add_child(action, child_state)
            self.search_stats['nodes_created'] += 1
            
            return child
            
        except Exception as e:
            error_msg = f"Node expansion failed: {e}"
            logger.error(error_msg)
            raise MCTSError(error_msg)
    
    def backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree"""
        try:
            current = node
            while current is not None:
                current.update(value)
                current = current.parent
                # Note: No alternating rewards since this is single-agent planning
                
        except Exception as e:
            error_msg = f"Backpropagation failed: {e}"
            logger.error(error_msg)
            raise MCTSError(error_msg)
    
    def search(self, root_state: State) -> ActionType:
        """
        Perform MCTS search and return best action
        
        Args:
            root_state: Current game state
            
        Returns:
            Best action according to MCTS
        """
        start_time = time.time()
        
        # Reset rollout summary for this search
        self.current_search_summary.reset()
        
        for attempt in range(self.max_retries):
            try:
                # Use consistent state hash format
                if getattr(self, 'use_optimized_state', True):
                    optimized_state = state_to_optimized_state(root_state)
                    state_hash = optimized_discrete_state_to_hash(optimized_state)
                else:
                    discrete_state = state_to_discrete_state(root_state)
                    state_hash = discrete_state_to_hash(discrete_state)
                
                if state_hash in self.node_cache:
                    root = self.node_cache[state_hash]
                    self.search_stats['cache_hits'] += 1
                else:
                    root = MCTSNode(root_state)
                    self.node_cache[state_hash] = root
                
                # MCTS iterations
                depths_reached = []
                for iteration in range(self.max_iterations):
                    # Check timeout
                    if time.time() - start_time > self.timeout_seconds:
                        self.search_stats['timeouts'] += 1
                        logger.warning(f"MCTS search timeout after {iteration} iterations")
                        break
                    
                    # Generate unique rollout ID for this iteration
                    rollout_id = f"search_{iteration:04d}"
                    
                    # Selection: traverse tree using UCB1
                    current = root
                    path = [current]
                    
                    while not current.is_terminal and current.is_fully_expanded():
                        current = self.select_child(current)
                        path.append(current)
                    
                    # Expansion: add new child node
                    if not current.is_terminal:
                        expanded_child = self.expand(current)
                        if expanded_child:
                            current = expanded_child
                            path.append(current)
                    
                    # Simulation: rollout from current node with FRESH intrinsic state and exploration tracking
                    if not current.is_terminal:
                        # Each simulation gets its own completely fresh intrinsic system and exploration tracking
                        simulation_value = self.simulate_with_fresh_intrinsics(
                            current.state, rollout_id, iteration, root_state, depth=0
                        )
                        self.search_stats['rollouts_performed'] += 1
                    else:
                        # Terminal node value
                        if current.state['state'] == 2:  # won
                            simulation_value = 10.0
                        elif current.state['state'] == 3:  # dead
                            simulation_value = -10.0
                        else:
                            simulation_value = 0.0
                    
                    # Add neural network value
                    _, network_value = self.get_network_prediction(current.state)
                    simulation_value = 0.7 * simulation_value + 0.3 * network_value
                    
                    # Backpropagation: update node values
                    self.backpropagate(current, simulation_value)
                    
                    # Track depth for statistics
                    depths_reached.append(len(path))
                
                # Select best action based on visit counts
                if not root.children:
                    # No children expanded, return safe action
                    valid_actions = get_contextual_actions(root_state['state'])
                    if not valid_actions:
                        raise MCTSError("No valid actions available")
                    return valid_actions[0]
                
                best_action = max(root.children.keys(), 
                                 key=lambda a: root.children[a].visit_count)
                
                # Update statistics
                search_time = time.time() - start_time
                self.search_stats['iterations_performed'] = len(depths_reached)
                self.search_stats['average_depth_reached'] = np.mean(depths_reached) if depths_reached else 0.0
                
                # Log rollout summary instead of individual rollouts
                summary = self.current_search_summary.get_summary()
                if summary["rollouts"] > 0:
                    logger.debug(f"MCTS search summary: {summary['rollouts']} rollouts, "
                                f"avg_reward={summary['avg_reward']:.3f}, "
                                f"reward_range=[{summary['reward_range'][0]:.3f}, {summary['reward_range'][1]:.3f}], "
                                f"avg_depth={summary['avg_depth']:.1f}, "
                                f"termination_reasons={summary['termination_reasons']}")
                
                logger.debug(f"MCTS search completed in {search_time:.3f}s, "
                            f"selected action: {best_action.name}, "
                            f"visits: {root.children[best_action].visit_count}, "
                            f"value: {root.children[best_action].mean_value:.3f}")

                # Training data collection and network training
                self.searches_since_training += 1

                # Calculate action probabilities directly from search results (avoid recursion)
                if root.children:
                    total_visits = sum(child.visit_count for child in root.children.values())
                    action_probs = {}
                    for action, child in root.children.items():
                        action_probs[action] = child.visit_count / total_visits if total_visits > 0 else 0.0
                else:
                    # Fallback for no children
                    valid_actions = get_contextual_actions(root_state['state'])
                    uniform_prob = 1.0 / len(valid_actions) if valid_actions else 0.0
                    action_probs = {action: uniform_prob for action in valid_actions}

                root_value = root.mean_value if root.children else 0.0
                self.collect_training_data(root_state, action_probs, root_value)

                # Train network periodically  
                if (self.searches_since_training >= self.train_every_n_searches and 
                    len(self.training_data['states']) >= self.batch_size):
                    logger.info(f"Triggering network training after {self.searches_since_training} searches")
                    self.train_network()
                    self.searches_since_training = 0

                return best_action
                
            except Exception as e:
                self.search_stats['search_failures'] += 1
                error_msg = f"MCTS search attempt {attempt + 1} failed: {e}"
                logger.error(error_msg)
                
                if attempt == self.max_retries - 1:
                    # Final attempt failed
                    logger.error("All MCTS search attempts failed!")
                    raise MCTSError(f"MCTS search failed after {self.max_retries} attempts: {e}")
                else:
                    logger.warning(f"Retrying MCTS search (attempt {attempt + 2}/{self.max_retries})")
                    time.sleep(0.1)  # Brief pause before retry
    
    def get_action_probabilities(self, root_state: State) -> Dict[ActionType, float]:
        """
        Get action probability distribution based on visit counts
        
        Args:
            root_state: Current game state
            
        Returns:
            Dictionary mapping actions to probabilities
        """
        try:
            # Use the correct state hash format based on configuration
            if getattr(self, 'use_optimized_state', True):
                # Use optimized state hash
                optimized_state = state_to_optimized_state(root_state)
                state_hash = optimized_discrete_state_to_hash(optimized_state)
            else:
                # Use legacy state hash
                discrete_state = state_to_discrete_state(root_state)
                state_hash = discrete_state_to_hash(discrete_state)
            
            # Check if we have this state in cache
            if state_hash not in self.node_cache:
                # Instead of calling search (which caused recursion), return uniform probabilities
                valid_actions = get_contextual_actions(root_state['state'])
                uniform_prob = 1.0 / len(valid_actions) if valid_actions else 0.0
                logger.warning(f"State not in cache, returning uniform probabilities for {len(valid_actions)} actions")
                return {action: uniform_prob for action in valid_actions}
            
            root = self.node_cache[state_hash]
            
            if not root.children:
                valid_actions = get_contextual_actions(root_state['state'])
                uniform_prob = 1.0 / len(valid_actions) if valid_actions else 0.0
                return {action: uniform_prob for action in valid_actions}
            
            # Calculate probabilities based on visit counts
            total_visits = sum(child.visit_count for child in root.children.values())
            probabilities = {}
            
            for action, child in root.children.items():
                probabilities[action] = child.visit_count / total_visits if total_visits > 0 else 0.0
            
            return probabilities
            
        except Exception as e:
            error_msg = f"Failed to get action probabilities: {e}"
            logger.error(error_msg)
            # Instead of raising an error, fall back to uniform distribution
            try:
                valid_actions = get_contextual_actions(root_state['state'])
                uniform_prob = 1.0 / len(valid_actions) if valid_actions else 0.0
                logger.warning(f"Falling back to uniform probabilities due to error: {e}")
                return {action: uniform_prob for action in valid_actions}
            except:
                raise MCTSError(error_msg)
    
    def collect_training_data(self, state: State, action_probs: Dict[ActionType, float], 
                            search_value: float):
        """Collect training data from MCTS search results (supports both state formats)"""
        if not self.learning_enabled:
            return
            
        if getattr(self, 'use_optimized_state', True):  # Default to optimized
            # Use optimized state representation
            optimized_state = state_to_optimized_state(state)
            state_tensor = self.encode_optimized_discrete_state(optimized_state)
        else:
            # Use legacy state representation
            discrete_state = state_to_discrete_state(state)
            state_tensor = self.encode_discrete_state(discrete_state)
        
        # Convert action probabilities to full vector
        action_prob_vector = torch.zeros(len(ActionType), device=self.device)
        for i, action in enumerate(ActionType):
            if action in action_probs:
                action_prob_vector[i] = action_probs[action]
        
        self.training_data['states'].append(state_tensor)
        self.training_data['action_probs'].append(action_prob_vector)
        self.training_data['values'].append(search_value)
        
        # Limit training data size
        max_data_size = self.mcts_config.get('max_training_data', 1000)
        if len(self.training_data['states']) > max_data_size:
            # Remove oldest data
            self.training_data['states'].pop(0)
            self.training_data['action_probs'].pop(0)
            self.training_data['values'].pop(0)
    
    def train_network(self):
        """Train the policy-value network using collected data"""
        if not self.learning_enabled or len(self.training_data['states']) < self.batch_size:
            return
        
        logger.info(f"Training network with {len(self.training_data['states'])} samples")
        
        # Store weights before training for change detection
        old_weights = {
            name: param.clone().detach() 
            for name, param in self.policy_value_net.named_parameters()
        }
        
        # Prepare training batch
        batch_size = min(self.batch_size, len(self.training_data['states']))
        indices = torch.randperm(len(self.training_data['states']))[:batch_size]
        
        batch_states = torch.stack([self.training_data['states'][i] for i in indices])
        batch_action_probs = torch.stack([self.training_data['action_probs'][i] for i in indices])
        batch_values = torch.tensor([self.training_data['values'][i] for i in indices], 
                                  dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Forward pass
        self.policy_value_net.train()
        pred_probs, pred_values = self.policy_value_net(batch_states)
        
        # Calculate losses
        policy_loss = F.cross_entropy(pred_probs, batch_action_probs)
        value_loss = F.mse_loss(pred_values, batch_values)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Calculate gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_value_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.policy_value_net.eval()
        
        # Track learning metrics
        self.learning_metrics['policy_loss_history'].append(policy_loss.item())
        self.learning_metrics['value_loss_history'].append(value_loss.item())
        self.learning_metrics['total_loss_history'].append(total_loss.item())
        self.learning_metrics['gradient_norms'].append(grad_norm.item())
        self.learning_metrics['training_count'] += 1
        self.learning_metrics['last_training_time'] = time.time()
        
        # Calculate weight changes
        weight_change = 0.0
        for name, param in self.policy_value_net.named_parameters():
            if name in old_weights:
                change = torch.norm(param - old_weights[name]).item()
                weight_change += change
        
        self.learning_metrics['weight_changes'].append(weight_change)
        
        # Log training progress
        logger.info(f"Training #{self.learning_metrics['training_count']}: "
                   f"Policy Loss: {policy_loss.item():.4f}, "
                   f"Value Loss: {value_loss.item():.4f}, "
                   f"Total Loss: {total_loss.item():.4f}, "
                   f"Grad Norm: {grad_norm.item():.4f}, "
                   f"Weight Change: {weight_change:.6f}")
        
        # Simple learning check
        self.check_learning_progress()
    
    def check_learning_progress(self):
        """Simple checks to verify the model is learning"""
        if self.learning_metrics['training_count'] < 5:
            return
        
        recent_losses = self.learning_metrics['total_loss_history'][-5:]
        weight_changes = self.learning_metrics['weight_changes'][-5:]
        grad_norms = self.learning_metrics['gradient_norms'][-5:]
        
        # Check 1: Are losses decreasing?
        loss_trend = recent_losses[-1] - recent_losses[0]
        if loss_trend < 0:
            logger.info("✓ Model learning: Loss is decreasing")
        else:
            logger.warning("⚠ Potential learning issue: Loss not decreasing")
        
        # Check 2: Are weights changing?
        avg_weight_change = sum(weight_changes) / len(weight_changes)
        if avg_weight_change > 1e-6:
            logger.info(f"✓ Model learning: Weights changing (avg: {avg_weight_change:.6f})")
        else:
            logger.warning(f"⚠ Potential learning issue: Weights barely changing (avg: {avg_weight_change:.6f})")
        
        # Check 3: Are gradients flowing?
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        if avg_grad_norm > 1e-4:
            logger.info(f"✓ Model learning: Gradients flowing (avg norm: {avg_grad_norm:.6f})")
        else:
            logger.warning(f"⚠ Potential learning issue: Very small gradients (avg norm: {avg_grad_norm:.6f})")
        
        # Check 4: Compare with initial weights
        if self.learning_metrics['training_count'] >= 10:
            total_change_from_init = 0.0
            for name, param in self.policy_value_net.named_parameters():
                if name in self.initial_weights:
                    change = torch.norm(param - self.initial_weights[name]).item()
                    total_change_from_init += change
            
            if total_change_from_init > 1e-3:
                logger.info(f"✓ Model learning: Significant change from initialization ({total_change_from_init:.6f})")
            else:
                logger.warning(f"⚠ Potential learning issue: Minimal change from initialization ({total_change_from_init:.6f})")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        stats = {
            'learning_enabled': self.learning_enabled,
            'training_data_size': len(self.training_data['states']),
            'training_count': self.learning_metrics['training_count'],
            'last_training_time': self.learning_metrics['last_training_time'],
            'searches_since_training': self.searches_since_training
        }
        
        if self.learning_metrics['training_count'] > 0:
            stats.update({
                'latest_policy_loss': self.learning_metrics['policy_loss_history'][-1],
                'latest_value_loss': self.learning_metrics['value_loss_history'][-1],
                'latest_total_loss': self.learning_metrics['total_loss_history'][-1],
                'latest_weight_change': self.learning_metrics['weight_changes'][-1],
                'latest_grad_norm': self.learning_metrics['gradient_norms'][-1],
                'avg_loss_last_5': sum(self.learning_metrics['total_loss_history'][-5:]) / min(5, len(self.learning_metrics['total_loss_history'])),
                'loss_trend_last_5': self.learning_metrics['total_loss_history'][-1] - self.learning_metrics['total_loss_history'][max(0, len(self.learning_metrics['total_loss_history'])-5)] if len(self.learning_metrics['total_loss_history']) >= 5 else 0.0
            })
        
        return stats
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Enhanced search statistics including learning metrics"""
        stats = {
            **self.search_stats,
            'cache_size': len(self.node_cache),
            'device': str(self.device),
            'network_enabled': self.policy_value_net is not None,
            'gpu_memory_allocated': torch.cuda.memory_allocated(self.device) / 1e9 if self.device.type == 'cuda' else 0.0,
            'gpu_memory_cached': torch.cuda.memory_reserved(self.device) / 1e9 if self.device.type == 'cuda' else 0.0,
            'intrinsics_per_rollout': 'fresh_isolated',
            'early_termination_enabled': self.enable_early_termination,
            'explorable_restriction_enabled': self.restrict_to_explorable,
            'exploration_tracking_enabled': self.enable_exploration_tracking and self.snapshot_manager is not None,
            'learning_stats': self.get_learning_statistics()
        }
        
        # Add last search summary
        if hasattr(self, 'current_search_summary'):
            summary = self.current_search_summary.get_summary()
            stats['last_search_summary'] = summary
        
        # Add termination breakdown
        if self.search_stats['early_terminations'] > 0:
            stats['termination_breakdown'] = {
                'total_early_terminations': self.search_stats['early_terminations'],
                'stagnation': self.search_stats['stagnation_terminations'],
                'hunger': self.search_stats['hunger_terminations'], 
                'goal_reached': self.search_stats['goal_terminations']
            }
        
        return stats
    
    def clear_cache(self):
        """Clear node cache to free memory"""
        self.node_cache.clear()
        self.search_stats['fresh_intrinsics_created'] = 0
        self.search_stats['rollout_contexts_created'] = 0
        self.search_stats['explorations_tracked'] = 0
        
        # Clear GPU cache if using GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        logger.info("MCTS node cache cleared")


def create_mcts_agent(config: Dict[str, Any], snapshot_manager=None) -> MCTS:
    """Create and initialize MCTS agent from configuration with optional snapshot manager"""
    try:
        return MCTS(config, snapshot_manager)
    except Exception as e:
        error_msg = f"Failed to create MCTS agent: {e}"
        logger.error(error_msg)
        raise MCTSError(error_msg)


# Test function
if __name__ == "__main__":
    # Test MCTS with sample configuration
    test_config = {
        'mcts': {
            'max_iterations': 10,  # Small for testing
            'max_depth': 5,
            'exploration_constant': 1.414,
            'simulation_steps': 3,
            'use_gpu': True,
            'require_gpu': True,
            'enable_exploration_tracking': True,
            'policy_value_net': {
                'enabled': True,
                'hidden_size': 64,
                'learning_rate': 0.001,
                'use_optimized_state': True # Added for testing optimized state
            },
            'intrinsics': {
                'fear_of_pain': {'weight': 1.0, 'enabled': True},
                'fear_of_hunger': {'weight': 0.8, 'enabled': True},
                'sense_of_progress': {'weight': 1.2, 'enabled': True}
            }
        }
    }
    
    # Create test state
    test_state = {
        "location": {"x": 0, "z": 0},
        "line_of_sight": [
            {"distance": 8.0, "type": 3},   # Trap nearby
            {"distance": 15.0, "type": 2},  # Checkpoint
            {"distance": 0.0, "type": 0}    # Empty
        ] + [{"distance": 0.0, "type": 0}] * 69,  # Fill to 72 rays
        "hitpoint": 100,
        "state": 0,  # normal
        "hunger": 45.0,
        "timestamp": 1234567890,
        "snapshot": {"data": [[0]], "width": 1, "height": 1, "resolution": 1.0,
                    "origin": {"x": 0, "z": 0}, "timestamp": 1234567890}
    }
    
    # Test MCTS
    try:
        print("Testing MCTS Agent with Exploration Tracking")
        print("=" * 60)
        
        # Create mock snapshot manager for testing
        from snapshot_manager import SnapshotManager
        snapshot_manager = SnapshotManager(debug_frames=True)
        
        mcts = create_mcts_agent(test_config, snapshot_manager)
        
        # Test multiple searches to see exploration tracking
        for i in range(2):
            print(f"\nSearch {i+1}:")
            best_action = mcts.search(test_state)
            stats = mcts.get_search_statistics()
            print(f"  Best action: {best_action.name}")
            print(f"  Fresh intrinsics created: {stats['fresh_intrinsics_created']}")
            print(f"  Explorations tracked: {stats['explorations_tracked']}")
            print(f"  Last search summary: {stats.get('last_search_summary', 'N/A')}")
        
        print(f"\nFinal statistics: {mcts.get_search_statistics()}")
        
    except MCTSError as e:
        print(f"MCTS Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)
