#!/usr/bin/env python3
"""
Test script to demonstrate state encoding functionality
"""

import json
from simsreal_types import (
    state_to_discrete_state, discrete_state_to_hash, 
    calculate_state_space_size, AgentStatus, HungerLevel, DistanceCategory
)
from stateencode import StateEncoder


def create_test_state():
    """Create a test state for encoding demonstration"""
    # Create line of sight with 72 rays (simulating Unity raycast)
    line_of_sight = []
    
    # Create some test rays with different objects
    for i in range(72):
        angle = i * 5  # 5 degrees per ray for 360° coverage
        
        # Add some obstacles in front (rays 0-10)
        if 0 <= i <= 10:
            line_of_sight.append({"distance": 8.0, "type": 1})  # obstacle, close
        # Add a goal at 45 degrees (ray 9)
        elif i == 9:
            line_of_sight.append({"distance": 15.0, "type": 4})  # goal, medium distance
        # Add food at 90 degrees (ray 18)
        elif i == 18:
            line_of_sight.append({"distance": 12.0, "type": 6})  # food, close-medium
        # Add trap at 180 degrees (ray 36)
        elif i == 36:
            line_of_sight.append({"distance": 25.0, "type": 3})  # trap, far
        # Add checkpoint at 270 degrees (ray 54)
        elif i == 54:
            line_of_sight.append({"distance": 35.0, "type": 2})  # checkpoint, too far
        # Most rays detect nothing (empty space)
        else:
            line_of_sight.append({"distance": 100.0, "type": 0})  # no object detected
    
    return {
        "location": {"x": 10, "z": -20},
        "line_of_sight": line_of_sight,
        "hitpoint": 85,
        "state": 0,  # normal
        "hunger": 45.0,  # medium hunger
        "timestamp": 1234567890,
        "snapshot": {
            "data": [[0]],
            "width": 1,
            "height": 1, 
            "resolution": 1.0,
            "origin": {"x": 0, "z": 0},
            "timestamp": 1234567890
        }
    }


def main():
    print("State Encoding Test")
    print("=" * 50)
    
    # Print state space statistics
    stats = calculate_state_space_size()
    print(f"State Space Analysis:")
    print(f"  Agent Status combinations: {stats['agent_status']}")
    print(f"  Hunger Level combinations: {stats['hunger_levels']}")
    print(f"  Per-direction combinations: {stats['per_direction']:,}")
    print(f"  Total state space: {stats['total_state_space']:,}")
    print(f"  Bits required: {stats['log2_total']}")
    print()
    
    # Create test state
    test_state = create_test_state()
    print("Test State Created:")
    print(f"  Agent position: ({test_state['location']['x']}, {test_state['location']['z']})")
    print(f"  Agent status: {test_state['state']} (0=normal)")
    print(f"  Hunger: {test_state['hunger']}")
    print(f"  Line of sight rays: {len(test_state['line_of_sight'])}")
    print()
    
    # Encode the state
    print("Encoding State...")
    discrete_state = state_to_discrete_state(test_state)
    
    print("Discrete State Result:")
    print(f"  Agent status: {discrete_state['agent_status']} ({AgentStatus(discrete_state['agent_status']).name})")
    print(f"  Hunger level: {discrete_state['hunger_level']} ({HungerLevel(discrete_state['hunger_level']).name})")
    print(f"  Number of ray directions: {len(discrete_state['rays'])}")
    print()
    
    # Analyze ray directions
    print("Ray Direction Analysis:")
    direction_names = ["FORWARD", "FORWARD_RIGHT", "RIGHT", "BACK_RIGHT", 
                      "BACK", "BACK_LEFT", "LEFT", "FORWARD_LEFT"]
    
    for i, ray in enumerate(discrete_state['rays']):
        objects = []
        if ray['has_obstacle']: objects.append("obstacle")
        if ray['has_checkpoint']: objects.append("checkpoint") 
        if ray['has_trap']: objects.append("trap")
        if ray['has_goal']: objects.append("goal")
        if ray['has_food']: objects.append("food")
        
        distance_name = DistanceCategory(ray['closest_distance']).name
        objects_str = ", ".join(objects) if objects else "none"
        
        print(f"  {direction_names[i]:>12}: {objects_str:>20} | closest: {distance_name}")
    
    # Generate hash
    state_hash = discrete_state_to_hash(discrete_state)
    print(f"\nState Hash: {state_hash}")
    print(f"Hash length: {len(state_hash)} characters")
    
    # Test encoder class
    print("\nTesting StateEncoder class...")
    encoder = StateEncoder(enable_logging=False)
    
    encoded_state = encoder.encode_state(test_state)
    generated_hash = encoder.get_state_hash(test_state)
    
    print(f"Encoder generated same result: {state_hash == generated_hash}")
    
    # Analyze ray distribution
    ray_analysis = encoder.analyze_ray_distribution(test_state['line_of_sight'])
    print(f"\nRay Distribution Analysis:")
    print(f"  Total rays: {ray_analysis['total_rays']}")
    print(f"  Relevant objects: {ray_analysis['relevant_objects']}")
    print(f"  Average distance: {ray_analysis['avg_distance']:.1f}")
    print(f"  Distance categories: {ray_analysis['distance_categories']}")


if __name__ == "__main__":
    main() 