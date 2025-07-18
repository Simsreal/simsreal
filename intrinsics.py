"""
Intrinsic motivation system for agent simulation and reward calculation.

Implements three core intrinsics:
1. Fear of Pain - Penalizes proximity to traps
2. Fear of Hunger - Penalizes based on hunger level  
3. Sense of Progress - Rewards reaching checkpoints and goals

Used for simulation-based planning and MCTS reward estimation.
Each instance is completely isolated to ensure independent rollout evaluation.
"""

import math
import time
import uuid
from typing import Dict, List, Any, Tuple
from simsreal_types import (
    State, DiscreteState, ActionType, Location, 
    ObjectType, DistanceCategory, HungerLevel, RayDirection,
    apply_action_to_location, apply_action_to_state,
    get_valid_actions, get_contextual_actions, state_to_discrete_state
)
from loguru import logger


class BaseIntrinsic:
    """Base class for all intrinsic motivations with proper isolation"""
    
    def __init__(self, name: str, weight: float = 1.0, instance_id: str = None):
        self.name = name
        self.weight = weight
        self.enabled = True
        self.instance_id = instance_id or str(uuid.uuid4())[:8]
        self.created_at = time.time()
        
        # Internal state that should be fresh for each instance
        self._internal_state = {}
        self._call_count = 0
        self._last_reward = 0.0
    
    def calculate_reward(self, state: State, action: ActionType, next_state: State = None, 
                        terminated: bool = False) -> float:
        """
        Calculate reward for this intrinsic given current state and action
        
        Args:
            state: Current state
            action: Action being taken
            next_state: Resulting state after action (optional)
            terminated: Whether the action led to termination
            
        Returns:
            Reward value (positive = good, negative = bad)
        """
        self._call_count += 1
        if not self.enabled:
            self._last_reward = 0.0
            return 0.0
        
        reward = self._calculate_reward_impl(state, action, next_state, terminated)
        self._last_reward = reward
        return reward
    
    def _calculate_reward_impl(self, state: State, action: ActionType, next_state: State = None, 
                              terminated: bool = False) -> float:
        """Implementation-specific reward calculation"""
        raise NotImplementedError("Subclasses must implement _calculate_reward_impl")
    
    def get_info(self) -> Dict[str, Any]:
        """Get intrinsic information for debugging"""
        return {
            "name": self.name,
            "weight": self.weight,
            "enabled": self.enabled,
            "instance_id": self.instance_id,
            "created_at": self.created_at,
            "call_count": self._call_count,
            "last_reward": self._last_reward
        }
    
    def reset_internal_state(self):
        """Reset any internal state for fresh evaluation"""
        self._internal_state.clear()
        self._call_count = 0
        self._last_reward = 0.0


class FearOfPain(BaseIntrinsic):
    """
    Fear of Pain intrinsic - penalizes proximity to traps
    
    Penalty zones:
    - 10+ units: No penalty
    - 5-10 units: Medium penalty
    - 0-5 units: Heavy penalty
    - Direct contact: Severe penalty
    - Stagnation: Heavy penalty (can't move due to obstacles)
    """
    
    def __init__(self, weight: float = 1.0, instance_id: str = None):
        super().__init__("FearOfPain", weight, instance_id)
        self.danger_zone_far = 10.0    # Start penalizing at 10 units
        self.danger_zone_close = 5.0   # Heavy penalty at 5 units
        self.penalty_far = -0.1        # Light penalty for far danger
        self.penalty_close = -0.5      # Heavy penalty for close danger
        self.penalty_contact = -2.0    # Severe penalty for contact
        self.penalty_stagnation = -1.0 # Heavy penalty for being stuck
        
        # Instance-specific state
        self._trap_encounters = 0
        self._closest_trap_distance = float('inf')
    
    def _calculate_reward_impl(self, state: State, action: ActionType, next_state: State = None, 
                              terminated: bool = False) -> float:
        # Use next_state if available, otherwise predict next position
        if next_state is not None:
            eval_location = next_state["location"]
            eval_state = next_state
        else:
            # Handle the new return format: (new_state, should_terminate)
            result = apply_action_to_state(state, action)
            if isinstance(result, tuple):
                eval_state, _ = result
            else:
                eval_state = result
            eval_location = eval_state["location"]
        
        total_penalty = 0.0
        closest_trap = float('inf')
        
        # Check all rays for traps
        for ray in state["line_of_sight"]:
            if ray["type"] == ObjectType.TRAP and ray["distance"] > 0:
                distance = ray["distance"]
                closest_trap = min(closest_trap, distance)
                self._trap_encounters += 1
                
                # Calculate penalty based on distance
                if distance <= self.danger_zone_close:
                    total_penalty += self.penalty_close
                elif distance <= self.danger_zone_far:
                    total_penalty += self.penalty_far
        
        # Update instance state
        if closest_trap != float('inf'):
            self._closest_trap_distance = closest_trap
        
        # Check if agent status indicates trap contact (fell_down = 1)
        if eval_state["state"] == 1:  # fell_down
            total_penalty += self.penalty_contact
        
        # Penalty for stagnation (terminated due to being stuck)
        if terminated and eval_location["x"] == state["location"]["x"] and eval_location["z"] == state["location"]["z"]:
            total_penalty += self.penalty_stagnation
        
        return total_penalty * self.weight
    
    def get_trap_distances(self, state: State) -> List[float]:
        """Get distances to all visible traps for debugging"""
        trap_distances = []
        for ray in state["line_of_sight"]:
            if ray["type"] == ObjectType.TRAP and ray["distance"] > 0:
                trap_distances.append(ray["distance"])
        return trap_distances
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "trap_encounters": self._trap_encounters,
            "closest_trap_distance": self._closest_trap_distance if self._closest_trap_distance != float('inf') else None
        })
        return info


class FearOfHunger(BaseIntrinsic):
    """
    Fear of Hunger intrinsic - penalizes based on hunger level
    
    Penalty based on discrete hunger levels:
    - LOW (0-30): Small penalty
    - MEDIUM (31-69): Medium penalty  
    - HIGH (70-100): Heavy penalty
    - Starvation (terminated due to hunger): Severe penalty
    """
    
    def __init__(self, weight: float = 1.0, instance_id: str = None):
        super().__init__("FearOfHunger", weight, instance_id)
        self.penalty_low = -0.05      # Small penalty for low hunger
        self.penalty_medium = -0.2    # Medium penalty for medium hunger
        self.penalty_high = -0.8      # Heavy penalty for high hunger
        self.penalty_starvation = -5.0 # Severe penalty for starvation
        
        # Instance-specific state
        self._hunger_history = []
        self._max_hunger_seen = 0.0
    
    def _calculate_reward_impl(self, state: State, action: ActionType, next_state: State = None, 
                              terminated: bool = False) -> float:
        # Use next_state hunger if available, otherwise current state
        if next_state is not None:
            hunger = next_state["hunger"]
        else:
            hunger = state["hunger"]
        
        # Track hunger for this instance
        self._hunger_history.append(hunger)
        self._max_hunger_seen = max(self._max_hunger_seen, hunger)
        
        # Severe penalty if terminated due to starvation
        if terminated and hunger <= 0:
            return self.penalty_starvation * self.weight
        
        # Convert to discrete hunger level
        if hunger <= 30:
            hunger_level = HungerLevel.LOW
        elif hunger <= 69:
            hunger_level = HungerLevel.MEDIUM
        else:
            hunger_level = HungerLevel.HIGH
        
        # Apply penalty based on hunger level
        if hunger_level == HungerLevel.LOW:
            penalty = self.penalty_low
        elif hunger_level == HungerLevel.MEDIUM:
            penalty = self.penalty_medium
        else:  # HIGH
            penalty = self.penalty_high
        
        return penalty * self.weight
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "max_hunger_seen": self._max_hunger_seen,
            "avg_hunger": sum(self._hunger_history) / len(self._hunger_history) if self._hunger_history else 0,
            "hunger_evaluations": len(self._hunger_history)
        })
        return info


class SenseOfProgress(BaseIntrinsic):
    """
    Sense of Progress intrinsic - rewards reaching checkpoints and goals
    
    Enhanced checkpoint rewards:
    - Seeing checkpoints: Small positive reward
    - Getting closer to checkpoints: Distance-based reward
    - Reaching checkpoints: Large reward
    - Visiting new checkpoints: Progression bonus
    - Seeing goals: Medium positive reward  
    - Reaching goals: Large reward
    - Standing up from fell_down: Recovery reward
    - Goal completion (won state): Huge reward
    """
    
    def __init__(self, weight: float = 1.0, instance_id: str = None):
        super().__init__("SenseOfProgress", weight, instance_id)
        
        # Enhanced checkpoint rewards
        self.reward_see_checkpoint = 0.2   # Increased reward for seeing checkpoint
        self.reward_approach_checkpoint = 0.5  # New: reward for getting closer
        self.reward_reach_checkpoint = 5.0  # Increased reward for reaching checkpoint
        self.reward_checkpoint_progression = 8.0  # New: bonus for visiting new checkpoints
        
        # Goal rewards (keeping existing balance)
        self.reward_see_goal = 0.3         # Slightly increased
        self.reward_reach_goal = 10.0      # Increased reward for reaching goal
        self.reward_goal_completion = 15.0 # Increased huge reward for winning
        
        # Other rewards
        self.reward_standup = 0.5          # Reward for recovering from fell_down
        
        # Distance thresholds
        self.proximity_threshold = 3.0     # Increased distance to consider "reached"
        self.approach_threshold = 15.0     # Distance to start giving approach rewards
        
        # Instance-specific state for checkpoint progression tracking
        self._checkpoints_seen = 0
        self._goals_seen = 0
        self._standups_performed = 0
        self._closest_checkpoint = float('inf')
        self._closest_goal = float('inf')
        self._previous_closest_checkpoint = float('inf')  # New: track distance changes
        self._checkpoints_reached = set()  # New: track which checkpoints were reached
        self._checkpoint_approach_count = 0  # New: count approach rewards
    
    def _calculate_reward_impl(self, state: State, action: ActionType, next_state: State = None, 
                              terminated: bool = False) -> float:
        total_reward = 0.0
        
        # Reward for standing up
        if action == ActionType.STANDUP and state["state"] == 1:  # fell_down
            total_reward += self.reward_standup
            self._standups_performed += 1
        
        # Huge reward for goal completion (terminated with won state)
        if terminated and next_state is not None and next_state["state"] == 2:  # won
            total_reward += self.reward_goal_completion
        
        # Enhanced checkpoint and goal detection with distance-based rewards
        current_closest_checkpoint = float('inf')
        checkpoint_rewards = 0.0
        goal_rewards = 0.0
        
        for ray in state["line_of_sight"]:
            if ray["distance"] > 0:
                if ray["type"] == ObjectType.CHECKPOINT:
                    distance = ray["distance"]
                    current_closest_checkpoint = min(current_closest_checkpoint, distance)
                    
                    # Basic seeing reward
                    checkpoint_rewards += self.reward_see_checkpoint
                    self._checkpoints_seen += 1
                    
                    # Distance-based approach reward
                    if distance <= self.approach_threshold:
                        # Stronger reward for being closer (inverse distance)
                        approach_reward = self.reward_approach_checkpoint * (1.0 - (distance / self.approach_threshold))
                        checkpoint_rewards += approach_reward
                        self._checkpoint_approach_count += 1
                    
                    # Large reward if very close (reached)
                    if distance <= self.proximity_threshold:
                        checkpoint_rewards += self.reward_reach_checkpoint
                        
                        # Track unique checkpoint reached (approximate by distance ranges)
                        checkpoint_id = f"cp_{int(distance * 10)}"  # Simple ID based on distance
                        if checkpoint_id not in self._checkpoints_reached:
                            self._checkpoints_reached.add(checkpoint_id)
                            checkpoint_rewards += self.reward_checkpoint_progression
                        
                elif ray["type"] == ObjectType.GOAL:
                    distance = ray["distance"]
                    goal_rewards += self.reward_see_goal
                    self._goals_seen += 1
                    self._closest_goal = min(self._closest_goal, distance)
                    
                    # Additional reward if very close (reached)
                    if distance <= self.proximity_threshold:
                        goal_rewards += self.reward_reach_goal
        
        # Reward for getting closer to checkpoints between actions
        if current_closest_checkpoint != float('inf'):
            self._closest_checkpoint = current_closest_checkpoint
            
            # Check if we got closer to a checkpoint
            if (self._previous_closest_checkpoint != float('inf') and 
                current_closest_checkpoint < self._previous_closest_checkpoint):
                distance_improvement = self._previous_closest_checkpoint - current_closest_checkpoint
                # Reward proportional to improvement, with diminishing returns
                improvement_reward = min(distance_improvement * 0.3, 1.0)
                checkpoint_rewards += improvement_reward
            
            self._previous_closest_checkpoint = current_closest_checkpoint
        
        total_reward += checkpoint_rewards + goal_rewards
        
        # Check if agent won (reached goal) in next state
        if next_state is not None and next_state["state"] == 2:  # won
            total_reward += self.reward_reach_goal
        
        return total_reward * self.weight
    
    def get_progress_info(self, state: State) -> Dict[str, Any]:
        """Get enhanced progress information for debugging"""
        checkpoints_seen = 0
        goals_seen = 0
        closest_checkpoint = float('inf')
        closest_goal = float('inf')
        checkpoints_in_range = 0
        
        for ray in state["line_of_sight"]:
            if ray["distance"] > 0:
                if ray["type"] == ObjectType.CHECKPOINT:
                    checkpoints_seen += 1
                    closest_checkpoint = min(closest_checkpoint, ray["distance"])
                    if ray["distance"] <= self.approach_threshold:
                        checkpoints_in_range += 1
                elif ray["type"] == ObjectType.GOAL:
                    goals_seen += 1
                    closest_goal = min(closest_goal, ray["distance"])
        
        return {
            "checkpoints_seen": checkpoints_seen,
            "checkpoints_in_range": checkpoints_in_range,
            "goals_seen": goals_seen,
            "closest_checkpoint": closest_checkpoint if closest_checkpoint != float('inf') else None,
            "closest_goal": closest_goal if closest_goal != float('inf') else None,
            "agent_status": state["state"],
            "checkpoints_reached_count": len(self._checkpoints_reached),
            "approach_threshold": self.approach_threshold,
            "proximity_threshold": self.proximity_threshold
        }
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "checkpoints_seen": self._checkpoints_seen,
            "goals_seen": self._goals_seen,
            "standups_performed": self._standups_performed,
            "closest_checkpoint": self._closest_checkpoint if self._closest_checkpoint != float('inf') else None,
            "closest_goal": self._closest_goal if self._closest_goal != float('inf') else None,
            "checkpoints_reached_unique": len(self._checkpoints_reached),
            "checkpoint_approach_rewards": self._checkpoint_approach_count
        })
        return info


class PenaltyForBackwardMovement(BaseIntrinsic):
    """
    Penalizes the agent for moving backward to encourage forward exploration.
    """
    
    def __init__(self, weight: float = 1.0, instance_id: str = None):
        super().__init__("PenaltyForBackwardMovement", weight, instance_id)
        self.penalty = -0.2  # Small penalty for moving backward

    def _calculate_reward_impl(self, state: State, action: ActionType, next_state: State = None, 
                              terminated: bool = False) -> float:
        
        if action == ActionType.MOVE_BACKWARD:
            return self.penalty * self.weight
        
        return 0.0


class CommonSense(BaseIntrinsic):
    """
    Penalizes illogical actions based on agent state.
    - Penalizes trying to stand up when already standing (state 0).
    - Penalizes not standing up when fallen down (state 1).
    """
    
    def __init__(self, weight: float = 1.0, instance_id: str = None):
        super().__init__("CommonSense", weight, instance_id)
        self.penalty_unnecessary_standup = -1.0 # Penalty for standing up when normal
        self.penalty_not_standing_up = -1.0   # Penalty for not standing up when fallen

    def _calculate_reward_impl(self, state: State, action: ActionType, next_state: State = None, 
                              terminated: bool = False) -> float:
        
        agent_status = state["state"]

        # Penalize trying to stand up when in a normal state
        if agent_status == 0 and action == ActionType.STANDUP:
            return self.penalty_unnecessary_standup * self.weight
            
        # Penalize doing anything other than standing up when fallen down
        if agent_status == 1 and action != ActionType.STANDUP:
            return self.penalty_not_standing_up * self.weight
            
        return 0.0


class FearOfDark(BaseIntrinsic):
    """
    Penalizes the agent for facing a blocked path, encouraging it to turn away from walls.
    A penalty is applied for each forward-facing ray that detects a very close object.
    """
    
    def __init__(self, weight: float = 1.0, instance_id: str = None):
        super().__init__("FearOfDark", weight, instance_id)
        self.penalty_per_blocked_ray = -0.3 # Penalty for each forward ray that is blocked

    def _calculate_reward_impl(self, state: State, action: ActionType, next_state: State = None, 
                              terminated: bool = False) -> float:
        
        # This intrinsic evaluates the current state, not the action's outcome.
        discrete_state = state_to_discrete_state(state)
        
        total_penalty = 0.0
        
        # Check forward-facing rays (FORWARD, FORWARD_LEFT, FORWARD_RIGHT)
        forward_directions = [RayDirection.FORWARD, RayDirection.FORWARD_LEFT, RayDirection.FORWARD_RIGHT]
        
        for ray in discrete_state['rays']:
            if ray['direction'] in forward_directions:
                if ray['closest_distance'] == DistanceCategory.VERY_CLOSE:
                    total_penalty += self.penalty_per_blocked_ray
                    
        return total_penalty * self.weight


class IntrinsicSystem:
    """
    Complete intrinsic motivation system that combines all intrinsics
    
    Each instance is completely isolated for independent MCTS rollout evaluation.
    Now handles termination signals properly.
    """
    
    def __init__(self, system_id: str = None):
        self.system_id = system_id or str(uuid.uuid4())[:8]
        self.created_at = time.time()
        
        # Create fresh intrinsic instances with unique IDs
        # Increased weight for sense_of_progress to encourage checkpoint seeking
        self.intrinsics = {
            "fear_of_pain": FearOfPain(weight=1.0, instance_id=f"{self.system_id}_pain"),
            "fear_of_hunger": FearOfHunger(weight=0.4, instance_id=f"{self.system_id}_hunger"),  # Reduced to prioritize checkpoints
            "sense_of_progress": SenseOfProgress(weight=3.0, instance_id=f"{self.system_id}_progress"),  # Increased weight
            "penalty_for_backward_movement": PenaltyForBackwardMovement(weight=0.2, instance_id=f"{self.system_id}_backward"),  # Reduced
            "common_sense": CommonSense(weight=1.5, instance_id=f"{self.system_id}_commonsense"),
            "fear_of_dark": FearOfDark(weight=0.8, instance_id=f"{self.system_id}_dark")  # Reduced to prioritize checkpoints
        }
        
        self.enabled = True
        self.debug_mode = False
        
        # System-level tracking
        self._total_reward_calculated = 0.0
        self._action_evaluations = 0
        
    def calculate_total_reward(self, state: State, action: ActionType, next_state: State = None) -> float:
        """
        Calculate total reward from all intrinsics (handles new apply_action_to_state format)
        
        Args:
            state: Current state
            action: Action being taken
            next_state: Resulting state after action (optional)
            
        Returns:
            Total reward value
        """
        if not self.enabled:
            return 0.0
        
        # Determine if this action leads to termination
        terminated = False
        actual_next_state = next_state
        
        if next_state is None:
            # Need to predict next state and check for termination
            result = apply_action_to_state(state, action)
            if isinstance(result, tuple):
                actual_next_state, terminated = result
            else:
                actual_next_state = result
                terminated = False
        
        total_reward = 0.0
        intrinsic_rewards = {}
        
        for name, intrinsic in self.intrinsics.items():
            reward = intrinsic.calculate_reward(state, action, actual_next_state, terminated)
            intrinsic_rewards[name] = reward
            total_reward += reward
        
        # Track system-level stats
        self._total_reward_calculated += total_reward
        self._action_evaluations += 1
        
        if self.debug_mode:
            termination_str = f", terminated={terminated}" if terminated else ""
            logger.debug(f"System {self.system_id}: Action {action.name} rewards: {intrinsic_rewards}, "
                        f"Total: {total_reward:.3f}{termination_str}")
        
        return total_reward
    
    def calculate_action_rewards(self, state: State) -> Dict[ActionType, float]:
        """
        Calculate rewards for all contextually valid actions from current state
        
        Args:
            state: Current state
            
        Returns:
            Dictionary mapping actions to their reward values
        """
        action_rewards = {}
        
        # Get contextually valid actions based on agent state
        valid_actions = get_contextual_actions(state["state"])
        
        for action in valid_actions:
            reward = self.calculate_total_reward(state, action)
            action_rewards[action] = reward
        
        return action_rewards
    
    def simulate_action_sequences(self, state: State, sequence_length: int = 3, 
                                num_sequences: int = 100) -> List[Tuple[List[ActionType], float]]:
        """
        Simulate multiple action sequences and return their total rewards
        (Now handles termination signals properly)
        
        Args:
            state: Starting state
            sequence_length: Length of action sequences to simulate
            num_sequences: Number of random sequences to try
            
        Returns:
            List of (action_sequence, total_reward) tuples
        """
        import random
        
        sequences = []
        
        for _ in range(num_sequences):
            # Generate random action sequence with contextual validity
            action_sequence = []
            current_state = state.copy()
            total_reward = 0.0
            
            for step in range(sequence_length):
                # Get valid actions for current state
                valid_actions = get_contextual_actions(current_state["state"])
                
                if not valid_actions:
                    break  # No valid actions available
                
                # Choose random valid action
                action = random.choice(valid_actions)
                action_sequence.append(action)
                
                # Calculate reward for this action
                reward = self.calculate_total_reward(current_state, action)
                total_reward += reward
                
                # Update state for next iteration (handle new format)
                result = apply_action_to_state(current_state, action)
                if isinstance(result, tuple):
                    current_state, terminated = result
                    if terminated:
                        break  # Sequence ended due to termination
                else:
                    current_state = result
            
            if action_sequence:  # Only add non-empty sequences
                sequences.append((action_sequence, total_reward))
        
        # Sort by reward (best first)
        sequences.sort(key=lambda x: x[1], reverse=True)
        return sequences
    
    def get_intrinsic_info(self) -> Dict[str, Any]:
        """Get information about all intrinsics for debugging"""
        info = {
            "system_id": self.system_id,
            "created_at": self.created_at,
            "enabled": self.enabled,
            "debug_mode": self.debug_mode,
            "total_reward_calculated": self._total_reward_calculated,
            "action_evaluations": self._action_evaluations,
            "intrinsics": {}
        }
        
        for name, intrinsic in self.intrinsics.items():
            info["intrinsics"][name] = intrinsic.get_info()
        
        return info
    
    def set_weights(self, weights: Dict[str, float]):
        """Set weights for intrinsics"""
        for name, weight in weights.items():
            if name in self.intrinsics:
                self.intrinsics[name].weight = weight
    
    def enable_intrinsic(self, name: str, enabled: bool = True):
        """Enable or disable specific intrinsic"""
        if name in self.intrinsics:
            self.intrinsics[name].enabled = enabled
    
    def reset_all_intrinsics(self):
        """Reset internal state of all intrinsics"""
        for intrinsic in self.intrinsics.values():
            intrinsic.reset_internal_state()
        
        self._total_reward_calculated = 0.0
        self._action_evaluations = 0


# Convenience functions for direct use
def create_intrinsic_system(system_id: str = None) -> IntrinsicSystem:
    """Create a fresh intrinsic system with unique ID"""
    return IntrinsicSystem(system_id)

def evaluate_action(state: State, action: ActionType, system_id: str = None) -> float:
    """Quick action evaluation using fresh intrinsic system"""
    system = create_intrinsic_system(system_id)
    return system.calculate_total_reward(state, action)

def get_best_actions(state: State, top_k: int = 3, system_id: str = None) -> List[Tuple[ActionType, float]]:
    """Get the top-k best actions for a given state using fresh system"""
    system = create_intrinsic_system(system_id)
    action_rewards = system.calculate_action_rewards(state)
    
    # Sort by reward (best first)
    sorted_actions = sorted(action_rewards.items(), key=lambda x: x[1], reverse=True)
    return sorted_actions[:top_k]


# Example usage and testing
if __name__ == "__main__":
    # Create test state with traps and checkpoints
    test_state = {
        "location": {"x": 0, "z": 0},
        "line_of_sight": [
            {"distance": 8.0, "type": ObjectType.TRAP},     # Trap at medium distance
            {"distance": 15.0, "type": ObjectType.CHECKPOINT}, # Checkpoint far away
            {"distance": 25.0, "type": ObjectType.GOAL},    # Goal very far
            {"distance": 100.0, "type": 0}                  # Empty space
        ],
        "hitpoint": 100,
        "state": 0,  # normal
        "hunger": 45.0,  # medium hunger
        "timestamp": 1234567890,
        "snapshot": {"data": [[0]], "width": 1, "height": 1, "resolution": 1.0, 
                    "origin": {"x": 0, "z": 0}, "timestamp": 1234567890}
    }
    
    print("Testing Intrinsic System with Termination Handling")
    print("=" * 50)
    
    # Test multiple isolated systems
    systems = []
    for i in range(3):
        system = create_intrinsic_system(f"test_{i}")
        system.debug_mode = True
        systems.append(system)
        
        print(f"\nSystem {i+1} (ID: {system.system_id}):")
        action_rewards = system.calculate_action_rewards(test_state)
        for action, reward in action_rewards.items():
            print(f"  {action.name}: {reward:.3f}")
    
    # Verify systems are isolated and termination handling works
    print("\nSystem isolation and termination verification:")
    for i, system in enumerate(systems):
        info = system.get_intrinsic_info()
        print(f"System {i+1} - Evaluations: {info['action_evaluations']}, "
              f"Total reward: {info['total_reward_calculated']:.3f}")
        
        for name, intrinsic_info in info["intrinsics"].items():
            print(f"  {name}: instance_id={intrinsic_info['instance_id']}, "
                  f"calls={intrinsic_info['call_count']}, last_reward={intrinsic_info['last_reward']:.3f}")
