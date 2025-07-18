"""
State encoding and discretization utilities for MCTS-optimized state representation.

Converts 72-ray continuous simulation states into discrete 8-direction symbolic 
representations suitable for efficient Monte Carlo Tree Search (MCTS).

Focus: obstacle, checkpoint, trap, goal, food (ignore unknown, people, explorable_area)
"""

import numpy as np
import uuid
import time
from typing import List, Dict, Any, Optional
from simsreal_types import (
    State, DiscreteState, DiscreteRayInfo, 
    AgentStatus, HungerLevel, DistanceCategory, ObjectType, RayDirection,
    discretize_hunger, discretize_distance, aggregate_rays_to_8_directions,
    state_to_discrete_state, discrete_state_to_hash, calculate_state_space_size
)
from loguru import logger


class StateEncoder:
    """
    Encodes simulation states into discrete symbolic representations for MCTS.
    
    Each encoder instance is completely isolated to ensure thread-safety
    and independent processing during MCTS rollouts.
    
    Key features:
    - 72 rays → 8 cardinal directions (45° each)
    - Focus on actionable objects: obstacle, checkpoint, trap, goal, food
    - Discrete hunger levels: low/medium/high
    - Discrete distances: very_close/close/medium/far/too_far
    - Agent status: normal vs fell_down (ignore terminal states won/dead)
    """
    
    def __init__(self, enable_logging: bool = True, encoder_id: str = None):
        self.encoder_id = encoder_id or str(uuid.uuid4())[:8]
        self.enable_logging = enable_logging
        self.created_at = time.time()
        
        # Calculate state space once
        self.state_space_stats = calculate_state_space_size()
        
        # Instance tracking
        self._encodings_performed = 0
        self._hashes_generated = 0
        self._last_encoded_hash = None
        
        if self.enable_logging:
            logger.debug(f"StateEncoder {self.encoder_id} initialized")
            logger.debug(f"State space: {self.state_space_stats['total_state_space']:,} combinations")
            logger.debug(f"Requires ~{self.state_space_stats['log2_total']} bits")
    
    def encode_state(self, state: State) -> DiscreteState:
        """
        Convert full simulation state to discrete MCTS state
        
        Args:
            state: Full simulation state
            
        Returns:
            Discrete state optimized for MCTS
        """
        try:
            discrete_state = state_to_discrete_state(state)
            self._encodings_performed += 1
            
            if self.enable_logging and self._encodings_performed % 100 == 0:
                logger.debug(f"Encoder {self.encoder_id}: {self._encodings_performed} encodings performed")
            
            return discrete_state
            
        except Exception as e:
            logger.error(f"Encoder {self.encoder_id}: Failed to encode state: {e}")
            raise
    
    def get_state_hash(self, state: State) -> str:
        """
        Generate compact hash string for MCTS node identification
        
        Args:
            state: Full simulation state
            
        Returns:
            Unique hash string for the discrete state
        """
        try:
            discrete_state = self.encode_state(state)
            state_hash = discrete_state_to_hash(discrete_state)
            
            self._hashes_generated += 1
            self._last_encoded_hash = state_hash
            
            return state_hash
            
        except Exception as e:
            logger.error(f"Encoder {self.encoder_id}: Failed to generate hash: {e}")
            raise
    
    def encode_state_for_network(self, state: State) -> np.ndarray:
        """
        Encode state into feature vector for neural network input
        
        Args:
            state: Full simulation state
            
        Returns:
            Numpy array ready for neural network
        """
        try:
            discrete_state = self.encode_state(state)
            features = []
            
            # Agent status (1 feature)
            features.append(float(discrete_state['agent_status']))
            
            # Hunger level (1 feature) 
            features.append(float(discrete_state['hunger_level']))
            
            # Ray information (8 directions * 8 features each = 64 features)
            for ray in discrete_state['rays']:
                features.extend([
                    float(ray['direction']),
                    float(ray['has_obstacle']),
                    float(ray['has_checkpoint']),
                    float(ray['has_trap']),
                    float(ray['has_goal']),
                    float(ray['has_food']),
                    float(ray['closest_distance']),
                    float(ray['closest_object_type'])
                ])
            
            # Pad rays to exactly 8 if needed
            while len(discrete_state['rays']) < 8:
                features.extend([0.0] * 8)  # Empty ray
            
            feature_array = np.array(features, dtype=np.float32)
            
            # Validate feature vector size
            expected_size = 1 + 1 + (8 * 8)  # 66 features total
            if len(feature_array) != expected_size:
                raise ValueError(f"Feature vector size mismatch: {len(feature_array)} != {expected_size}")
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Encoder {self.encoder_id}: Failed to encode for network: {e}")
            raise
    
    def analyze_ray_distribution(self, line_of_sight: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze raycast data for debugging and environment understanding"""
        if not line_of_sight:
            return {'error': 'No raycast data', 'encoder_id': self.encoder_id}
        
        # Count relevant object types (1,2,3,4,6)
        relevant_objects = {1: 0, 2: 0, 3: 0, 4: 0, 6: 0}  # obstacle, checkpoint, trap, goal, food
        distances = []
        
        for ray in line_of_sight:
            obj_type = ray.get('type', 0)
            distance = ray.get('distance', 0)
            
            if obj_type in relevant_objects:
                relevant_objects[obj_type] += 1
            
            if distance > 0:
                distances.append(distance)
        
        # Distance categorization
        distance_categories = {cat.name: 0 for cat in DistanceCategory}
        for distance in distances:
            cat = discretize_distance(distance)
            cat_name = DistanceCategory(cat).name
            distance_categories[cat_name] += 1
        
        return {
            'encoder_id': self.encoder_id,
            'total_rays': len(line_of_sight),
            'relevant_objects': relevant_objects,
            'distance_categories': distance_categories,
            'avg_distance': np.mean(distances) if distances else 0,
            'total_relevant_objects': sum(relevant_objects.values())
        }
    
    def get_encoding_statistics(self) -> Dict[str, Any]:
        """Get encoding performance statistics"""
        return {
            'encoder_id': self.encoder_id,
            'created_at': self.created_at,
            'encodings_performed': self._encodings_performed,
            'hashes_generated': self._hashes_generated,
            'last_encoded_hash': self._last_encoded_hash,
            'state_space_size': self.state_space_stats['total_state_space'],
            'bits_required': self.state_space_stats['log2_total']
        }
    
    def validate_state_consistency(self, state1: State, state2: State) -> bool:
        """
        Validate that two states produce the same hash if they're equivalent
        
        Args:
            state1: First state
            state2: Second state
            
        Returns:
            True if states produce same hash, False otherwise
        """
        try:
            hash1 = self.get_state_hash(state1)
            hash2 = self.get_state_hash(state2)
            return hash1 == hash2
        except Exception as e:
            logger.error(f"Encoder {self.encoder_id}: State consistency validation failed: {e}")
            return False


# Convenience functions for direct use
def encode_state_for_mcts(state: State, encoder_id: str = None) -> DiscreteState:
    """Quick state encoding for MCTS with optional encoder ID"""
    encoder = StateEncoder(enable_logging=False, encoder_id=encoder_id)
    return encoder.encode_state(state)

def get_mcts_hash(state: State, encoder_id: str = None) -> str:
    """Quick hash generation for MCTS with optional encoder ID"""
    encoder = StateEncoder(enable_logging=False, encoder_id=encoder_id)
    return encoder.get_state_hash(state)

def create_encoder_for_rollout(rollout_id: str) -> StateEncoder:
    """Create isolated encoder for specific rollout"""
    return StateEncoder(enable_logging=False, encoder_id=f"rollout_{rollout_id}")

if __name__ == "__main__":
    print("MCTS State Encoding Isolation Test")
    print("=" * 40)
    
    # Create test state
    test_state = {
        "location": {"x": 5, "z": 10},
        "line_of_sight": [
            {"distance": 8.0, "type": 3},   # Trap
            {"distance": 15.0, "type": 2},  # Checkpoint
            {"distance": 0.0, "type": 0}    # Empty
        ] + [{"distance": 0.0, "type": 0}] * 69,
        "hitpoint": 100,
        "state": 0,
        "hunger": 45.0,
        "timestamp": 1234567890,
        "snapshot": {"data": [[0]], "width": 1, "height": 1, "resolution": 1.0,
                    "origin": {"x": 0, "z": 0}, "timestamp": 1234567890}
    }
    
    # Test multiple isolated encoders
    encoders = []
    for i in range(3):
        encoder = StateEncoder(enable_logging=True, encoder_id=f"test_{i}")
        encoders.append(encoder)
        
        # Test encoding
        discrete_state = encoder.encode_state(test_state)
        state_hash = encoder.get_state_hash(test_state)
        
        print(f"\nEncoder {i+1} (ID: {encoder.encoder_id}):")
        print(f"  Hash: {state_hash[:20]}...")
        print(f"  Agent status: {discrete_state['agent_status']}")
        print(f"  Hunger level: {discrete_state['hunger_level']}")
        print(f"  Ray count: {len(discrete_state['rays'])}")
    
    # Verify all encoders produce same hash for same input
    hashes = [encoder.get_state_hash(test_state) for encoder in encoders]
    all_same = all(h == hashes[0] for h in hashes)
    print(f"\nHash consistency across encoders: {'PASS' if all_same else 'FAIL'}")
    
    # Show encoding statistics
    print("\nEncoder statistics:")
    for i, encoder in enumerate(encoders):
        stats = encoder.get_encoding_statistics()
        print(f"  Encoder {i+1}: {stats['encodings_performed']} encodings, "
              f"{stats['hashes_generated']} hashes")
    
    # Print simplified state space analysis
    stats = calculate_state_space_size()
    print(f"\nState Space Analysis:")
    print(f"  Total combinations: {stats['total_state_space']:,}")
    print(f"  Bits required: {stats['log2_total']}")
