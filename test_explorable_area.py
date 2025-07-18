#!/usr/bin/env python3
"""
Test script to generate sample frames showing explorable area visualization
"""

import json
import os
import numpy as np
from snapshot_manager import SnapshotManager
from simsreal_types import State, LineOfSight

def create_test_scenario(scenario_name: str, line_of_sight_data: list) -> State:
    """Create a test state with given line of sight data"""
    return {
        "location": {"x": 0, "z": 0},
        "line_of_sight": line_of_sight_data,
        "hitpoint": 100,
        "state": 1,
        "hunger": 50.0,
        "timestamp": 1234567890
    }

def main():
    # Initialize snapshot manager with debug frames enabled
    snapshot_manager = SnapshotManager(base_dir="test_snapshots", debug_frames=True)
    
    # Test Scenario 1: Mixed obstacles and free space
    print("Generating test scenario 1: Mixed environment...")
    scenario1_los = []
    num_rays = 36  # 10 degree increments for 360 degrees
    
    for i in range(num_rays):
        angle = i * 10  # degrees
        distance = 50.0  # default max range
        obj_type = 0    # default no object
        
        # Create some obstacles at specific angles
        if 80 <= angle <= 120:  # Obstacle wall ahead-right
            distance = 20.0
            obj_type = 1  # obstacle
        elif 200 <= angle <= 220:  # Small obstacle behind-left
            distance = 15.0
            obj_type = 1  # obstacle
        elif angle == 45:  # Goal at 45 degrees
            distance = 30.0
            obj_type = 4  # goal
        elif angle == 315:  # Food at 315 degrees
            distance = 25.0
            obj_type = 6  # food
        elif 260 <= angle <= 280:  # Checkpoint behind
            distance = 35.0
            obj_type = 2  # checkpoint
        
        scenario1_los.append({
            "distance": distance,
            "type": obj_type
        })
    
    state1 = create_test_scenario("mixed_environment", scenario1_los)
    snapshot_manager.save_snapshot(state1)
    
    # Test Scenario 2: Corridor scenario
    print("Generating test scenario 2: Corridor...")
    scenario2_los = []
    for i in range(num_rays):
        angle = i * 10
        distance = 50.0
        obj_type = 0
        
        # Create corridor walls on left and right
        if 70 <= angle <= 110 or 250 <= angle <= 290:  # Left and right walls
            distance = 10.0
            obj_type = 1  # obstacle
        elif angle == 0:  # Goal straight ahead
            distance = 40.0
            obj_type = 4  # goal
        
        scenario2_los.append({
            "distance": distance,
            "type": obj_type
        })
    
    state2 = create_test_scenario("corridor", scenario2_los)
    snapshot_manager.save_snapshot(state2)
    
    # Test Scenario 3: Open area with scattered objects
    print("Generating test scenario 3: Open area...")
    scenario3_los = []
    for i in range(num_rays):
        angle = i * 10
        distance = 50.0
        obj_type = 0
        
        # Scattered objects
        if angle == 30:  # Trap
            distance = 20.0
            obj_type = 3
        elif angle == 150:  # People
            distance = 25.0
            obj_type = 5
        elif angle == 210:  # Food
            distance = 15.0
            obj_type = 6
        elif angle == 300:  # Obstacle
            distance = 35.0
            obj_type = 1
        
        scenario3_los.append({
            "distance": distance,
            "type": obj_type
        })
    
    state3 = create_test_scenario("open_area", scenario3_los)
    snapshot_manager.save_snapshot(state3)
    
    # Test Scenario 4: Enclosed area (mostly obstacles)
    print("Generating test scenario 4: Enclosed area...")
    scenario4_los = []
    for i in range(num_rays):
        angle = i * 10
        distance = 12.0  # Close walls
        obj_type = 1     # Mostly obstacles
        
        # Create a few openings
        if 80 <= angle <= 100:  # Opening to the right
            distance = 50.0
            obj_type = 0
        elif 170 <= angle <= 190:  # Small opening behind
            distance = 30.0
            obj_type = 0
        elif angle == 90:  # Goal in the opening
            distance = 35.0
            obj_type = 4
        
        scenario4_los.append({
            "distance": distance,
            "type": obj_type
        })
    
    state4 = create_test_scenario("enclosed_area", scenario4_los)
    snapshot_manager.save_snapshot(state4)
    
    print(f"\nTest frames generated in: {snapshot_manager.current_run_dir}")
    print("Files created:")
    for file in os.listdir(snapshot_manager.current_run_dir):
        if file.endswith('.png'):
            print(f"  - {file}")

if __name__ == "__main__":
    main() 