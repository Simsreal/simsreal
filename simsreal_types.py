from typing import TypedDict, List
from enum import IntEnum

class Location(TypedDict):
    """
    x: int - X coordinate in the world
    z: int - Z coordinate in the world (forward/back in Unity)
    """
    x: int
    z: int

class LineOfSight(TypedDict):
    """
    distance: float - Distance to detected object (0 if no object detected)
    type: int - Object type: 0=unknown, 1=obstacle, 2=checkpoint, 3=trap, 4=goal, 5=people, 6=food
    """
    distance: float
    type: int

class State(TypedDict):
    """
    location: Location - Agent's position in the world
    line_of_sight: List[LineOfSight] - Objects detected by raycasting
    hitpoint: int - Agent's current hit points
    state: int - Agent status: 0=normal, 1=fell_down, 2=won, 3=dead
    hunger: float - Hunger level (0-100)
    timestamp: int - Timestamp in milliseconds since Unix epoch (UTC)
    snapshot: Snapshot - Constructed 2D map snapshot from current sensor data
    """
    location: Location
    line_of_sight: List[LineOfSight]
    hitpoint: int
    state: int
    hunger: float # 0-100
    timestamp: int
    snapshot: 'Snapshot'

class Action(TypedDict):
    """
    movement: str - Movement action: "moveforward", "movebackward", "moveleft", "moveright", "standup"
    confidence: float - Confidence level (0.0-1.0)
    """
    movement: str
    confidence: float

class Command(TypedDict):
    """
    timestamp: int - Command timestamp
    action: Action - Action to execute
    """
    timestamp: int
    action: Action 

class Snapshot(TypedDict):
    """
    data: List[List[int]] - Snapshot data (2D grid with object types)
    width: int - Width of the map
    height: int - Height of the map
    resolution: float - Resolution of the map (meters per pixel)
    origin: Location - Origin of the map in world coordinates
    timestamp: int - Timestamp of the map
    """
    data: List[List[int]]
    width: int
    height: int
    resolution: float
    origin: Location
    timestamp: int

class SnapshotHistory(TypedDict):
    """
    data: List[Snapshot] - List of snapshots
    """
    data: List[Snapshot]

# ============================================================================
# SIMULATION AND EARLY TERMINATION UTILITIES
# ============================================================================

def is_position_explorable(location: Location, snapshot: Snapshot) -> bool:
    """
    Check if a position is within explorable area (type 7) in the snapshot
    
    Args:
        location: World coordinates to check
        snapshot: Current snapshot data
        
    Returns:
        True if position is explorable, False otherwise
    """
    # Convert world coordinates to snapshot grid coordinates
    origin_x = snapshot["origin"]["x"]
    origin_z = snapshot["origin"]["z"]
    resolution = snapshot["resolution"]
    
    grid_x = int((location["x"] - origin_x) / resolution)
    grid_z = int((location["z"] - origin_z) / resolution)
    
    # Check bounds
    if (0 <= grid_x < snapshot["width"] and 
        0 <= grid_z < snapshot["height"]):
        
        # Check if this position is explorable (type 7) or empty (type 0)
        cell_type = snapshot["data"][grid_z][grid_x]
        return cell_type in [0, 7]  # Allow movement in empty or explorable areas
    
    return False  # Out of bounds

def check_goal_reached(state: State) -> bool:
    """
    Check if agent has reached a goal
    
    Args:
        state: Current state
        
    Returns:
        True if goal reached
    """
    # Check if agent state indicates goal reached
    if state["state"] == 2:  # won state
        return True
    
    # Check if very close to a goal object
    for ray in state["line_of_sight"]:
        if ray["type"] == 4 and ray["distance"] > 0 and ray["distance"] <= 2.0:  # Goal within 2 units
            return True
    
    return False

def simulate_hunger_decay(initial_hunger: float, elapsed_seconds: float) -> float:
    """
    Simulate hunger decay over time
    
    Args:
        initial_hunger: Starting hunger level (0-100)
        elapsed_seconds: Time elapsed in seconds
        
    Returns:
        New hunger level after decay
    """
    # Decay 1 point per second
    new_hunger = initial_hunger - elapsed_seconds
    return max(0.0, new_hunger)  # Don't go below 0

def check_stagnation(old_location: Location, new_location: Location) -> bool:
    """
    Check if agent is stagnating (not moving)
    
    Args:
        old_location: Previous location
        new_location: New location after action
        
    Returns:
        True if agent hasn't moved
    """
    return (old_location["x"] == new_location["x"] and 
            old_location["z"] == new_location["z"])

# ============================================================================
# VALID ACTIONS FOR SIMULATION (matching Unity AgentController.cs)
# ============================================================================

class ActionType(IntEnum):
    """Valid movement actions that match Unity AgentController commands"""
    MOVE_FORWARD = 0   # moveforward - forward direction (+Z in Unity)
    MOVE_BACKWARD = 1  # movebackward - backward direction (-Z in Unity)
    MOVE_LEFT = 2      # moveleft - left direction (-X in Unity)
    MOVE_RIGHT = 3     # moveright - right direction (+X in Unity)
    STANDUP = 4        # standup - recover from fell_down state

# Action name mappings for Unity compatibility
ACTION_NAMES = {
    ActionType.MOVE_FORWARD: "moveforward", 
    ActionType.MOVE_BACKWARD: "movebackward",
    ActionType.MOVE_LEFT: "moveleft",
    ActionType.MOVE_RIGHT: "moveright",
    ActionType.STANDUP: "standup"
}

# Reverse mapping for parsing
NAME_TO_ACTION = {v: k for k, v in ACTION_NAMES.items()}

# Movement deltas in Unity coordinate system (1 unit per move)
# Note: STANDUP doesn't change position, just changes agent state
ACTION_DELTAS = {
    ActionType.MOVE_FORWARD: (0, 1),   # +Z (forward)
    ActionType.MOVE_BACKWARD: (0, -1), # -Z (backward)
    ActionType.MOVE_LEFT: (-1, 0),     # -X (left)
    ActionType.MOVE_RIGHT: (1, 0),     # +X (right)
    ActionType.STANDUP: (0, 0)         # No movement, just state change
}

def get_valid_actions() -> List[ActionType]:
    """Get list of all valid actions for simulation"""
    return list(ActionType)

def get_movement_actions() -> List[ActionType]:
    """Get list of actions that actually move the agent"""
    return [ActionType.MOVE_FORWARD, ActionType.MOVE_BACKWARD, 
            ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT]

def get_contextual_actions(agent_state: int) -> List[ActionType]:
    """
    Get valid actions based on agent state
    
    Args:
        agent_state: 0=normal, 1=fell_down, 2=won, 3=dead
        
    Returns:
        List of valid actions for current state
    """
    if agent_state == 1:  # fell_down
        return [ActionType.STANDUP]  # Only standup is valid
    elif agent_state in [0]:  # normal
        return get_movement_actions()  # All movement actions
    else:  # won or dead (terminal states)
        return []  # No actions available

def action_to_delta(action: ActionType) -> tuple:
    """Convert action to (x, z) coordinate delta"""
    return ACTION_DELTAS[action]

def apply_action_to_location(location: Location, action: ActionType) -> Location:
    """Apply action to location and return new location"""
    dx, dz = action_to_delta(action)
    return {
        "x": location["x"] + dx,
        "z": location["z"] + dz
    }

def apply_action_to_state(state: State, action: ActionType, 
                         restrict_to_explorable: bool = True,
                         simulate_time: bool = True, 
                         time_step_seconds: float = 1.0) -> tuple[State, bool]:
    """
    Apply action to state and return new state with early termination check
    
    Args:
        state: Current state
        action: Action to apply
        restrict_to_explorable: Whether to restrict movement to explorable areas
        simulate_time: Whether to simulate hunger decay
        time_step_seconds: Time step for simulation
        
    Returns:
        (new_state, should_terminate) tuple
    """
    new_state = state.copy()
    should_terminate = False
    
    # Calculate new location for movement actions
    if action in get_movement_actions():
        new_location = apply_action_to_location(state["location"], action)
        
        # Check if new location is explorable
        if restrict_to_explorable:
            if is_position_explorable(new_location, state["snapshot"]):
                new_state["location"] = new_location
            else:
                # Movement blocked - stay in same location (stagnation)
                new_location = state["location"]
                new_state["location"] = new_location
        else:
            new_state["location"] = new_location
        
        # Check for stagnation
        if check_stagnation(state["location"], new_state["location"]):
            should_terminate = True  # Stagnation detected
    
    # Update agent status for standup action
    elif action == ActionType.STANDUP and state["state"] == 1:  # fell_down
        new_state["state"] = 0  # normal
    
    # Simulate hunger decay if requested
    if simulate_time:
        new_hunger = simulate_hunger_decay(state["hunger"], time_step_seconds)
        new_state["hunger"] = new_hunger
        
        # Check hunger termination
        if new_hunger <= 0:
            should_terminate = True
    
    # Check goal reached
    if check_goal_reached(new_state):
        should_terminate = True
        new_state["state"] = 2  # won
    
    return new_state, should_terminate

# Legacy function for backward compatibility
def apply_action_to_state_legacy(state: State, action: ActionType) -> State:
    """Legacy function - returns only the new state without termination check"""
    new_state, _ = apply_action_to_state(state, action, 
                                       restrict_to_explorable=False, 
                                       simulate_time=False)
    return new_state

# ============================================================================
# DISCRETE STATE TYPES FOR MCTS EFFICIENCY
# ============================================================================

class AgentStatus(IntEnum):
    """Discrete agent status values (ignoring hitpoint)"""
    NORMAL = 0
    FELL_DOWN = 1  # Only care about fell_down vs normal for MCTS

class HungerLevel(IntEnum):
    """Discrete hunger levels for MCTS state space reduction"""
    LOW = 0        # 0-30: Not hungry
    MEDIUM = 1     # 31-69: Moderately hungry  
    HIGH = 2       # 70-100: Very hungry

class DistanceCategory(IntEnum):
    """Discrete distance categories for raycast objects"""
    VERY_CLOSE = 0   # 0-5: Extremely close
    CLOSE = 1        # 5-10: Close
    MEDIUM = 2       # 10-20: Medium distance
    FAR = 3          # 20-30: Far
    TOO_FAR = 4      # 30+: Too far

class ObjectType(IntEnum):
    """Relevant object types for MCTS (excluding unknown, people)"""
    OBSTACLE = 1     # Blocks movement
    CHECKPOINT = 2   # Progress points
    TRAP = 3         # Harmful objects
    GOAL = 4         # Target objectives
    FOOD = 6         # Reduces hunger

class RayDirection(IntEnum):
    """8 cardinal and diagonal directions for ray aggregation"""
    FORWARD = 0        # 0° (North)
    FORWARD_RIGHT = 1  # 45° (Northeast)
    RIGHT = 2          # 90° (East)
    BACK_RIGHT = 3     # 135° (Southeast)
    BACK = 4           # 180° (South)
    BACK_LEFT = 5      # 225° (Southwest)
    LEFT = 6           # 270° (West)
    FORWARD_LEFT = 7   # 315° (Northwest)

class DiscreteRayInfo(TypedDict):
    """
    Discrete representation of raycast information for a single direction
    """
    direction: int  # RayDirection enum value
    has_obstacle: bool
    has_checkpoint: bool
    has_trap: bool
    has_goal: bool
    has_food: bool
    closest_distance: int  # DistanceCategory enum value
    closest_object_type: int  # ObjectType enum value (0 if no object)

class DiscreteState(TypedDict):
    """
    Compact discrete state representation optimized for MCTS
    - Agent status: 2 values (normal vs fell_down, ignore won/dead as terminal)
    - Hunger level: 3 values (low/medium/high)
    - Ray info: 8 directions with 5 object types each
    """
    agent_status: int  # AgentStatus enum value (0=normal, 1=fell_down)
    hunger_level: int  # HungerLevel enum value
    rays: List[DiscreteRayInfo]  # 8 directions worth of ray information
    timestamp: int

# ============================================================================
# OPTIMIZED DISCRETE STATE TYPES FOR EFFICIENT LEARNING
# ============================================================================

class OptimizedDistanceCategory(IntEnum):
    """Simplified distance categories - reduced from 5 to 3"""
    CLOSE = 0      # 0-10: Within interaction range
    MEDIUM = 1     # 10-25: Approaching range  
    FAR = 2        # 25+: Distant

class ImportantObjectType(IntEnum):
    """Consolidated object types for learning efficiency"""
    NONE = 0           # No important object
    OBSTACLE = 1       # Blocks movement (was ObjectType.OBSTACLE)
    PROGRESS = 2       # Checkpoints and goals combined (was ObjectType.CHECKPOINT + GOAL)
    DANGER = 3         # Traps (was ObjectType.TRAP)
    RESOURCE = 4       # Food (was ObjectType.FOOD)

class DirectionalZone(IntEnum):
    """Reduced from 8 to 4 cardinal directions for simplicity"""
    FORWARD = 0    # Front (0° ± 45°)
    RIGHT = 1      # Right (90° ± 45°) 
    BACKWARD = 2   # Back (180° ± 45°)
    LEFT = 3       # Left (270° ± 45°)

class OptimizedRayInfo(TypedDict):
    """
    Simplified ray information - reduced from 8 to 4 features per direction
    State space per direction: 4 × 5 × 3 = 60 combinations (vs 960 before)
    """
    zone: int                    # DirectionalZone enum
    has_obstacle: bool          # True if any obstacle in this zone
    important_object: int       # ImportantObjectType enum (most important object)
    closest_distance: int       # OptimizedDistanceCategory enum

class SpatialContext(TypedDict):
    """
    Coarse spatial position encoding to reduce state aliasing
    Uses grid-based hashing for approximate position awareness
    """
    grid_x: int        # Coarse X position (0-7 representing 8x8 grid)
    grid_z: int        # Coarse Z position (0-7 representing 8x8 grid) 
    facing: int        # Approximate facing direction (0-3 for N/E/S/W)

class OptimizedDiscreteState(TypedDict):
    """
    Highly optimized discrete state for efficient learning
    
    Total features: 20 (vs 66 before)
    - Agent status: 1 feature
    - Hunger level: 1 feature  
    - Spatial context: 3 features
    - Ray info: 16 features (4 zones × 4 features each)
    
    Theoretical state space: ~10^12 (vs 10^24 before) - 12 orders of magnitude reduction!
    """
    agent_status: int           # AgentStatus enum (0=normal, 1=fell_down)
    hunger_level: int          # HungerLevel enum (0=low, 1=medium, 2=high)
    spatial_context: SpatialContext  # Coarse position and orientation
    zones: List[OptimizedRayInfo]    # 4 directional zones
    timestamp: int

# ============================================================================
# UTILITY FUNCTIONS FOR DISCRETIZATION
# ============================================================================

def discretize_hunger(hunger: float) -> int:
    """Convert continuous hunger (0-100) to discrete levels"""
    if hunger <= 30:
        return HungerLevel.LOW
    elif hunger <= 69:
        return HungerLevel.MEDIUM
    else:
        return HungerLevel.HIGH

def discretize_distance(distance: float) -> int:
    """Convert continuous distance to discrete categories"""
    if distance < 5:
        return DistanceCategory.VERY_CLOSE
    elif distance < 10:
        return DistanceCategory.CLOSE
    elif distance < 20:
        return DistanceCategory.MEDIUM
    elif distance < 30:
        return DistanceCategory.FAR
    else:
        return DistanceCategory.TOO_FAR

def aggregate_rays_to_8_directions(line_of_sight: List[LineOfSight]) -> List[DiscreteRayInfo]:
    """
    Aggregate 72 rays into 8 cardinal/diagonal directions
    Each direction covers 45° arc (9 rays per direction)
    Ignore types: 0=unknown, 5=people, 7=explorable_area
    """
    num_rays = len(line_of_sight)
    if num_rays == 0:
        return []
    
    # Initialize 8 directions
    direction_info = []
    rays_per_direction = max(1, num_rays // 8)
    
    for direction in range(8):
        # Calculate which rays belong to this direction
        start_ray = direction * rays_per_direction
        end_ray = min(start_ray + rays_per_direction, num_rays)
        
        # Aggregate rays in this direction
        has_obstacle = False
        has_checkpoint = False
        has_trap = False
        has_goal = False
        has_food = False
        closest_distance = DistanceCategory.TOO_FAR
        closest_object_type = 0
        min_distance = float('inf')
        
        for ray_idx in range(start_ray, end_ray):
            if ray_idx >= len(line_of_sight):
                break
                
            ray = line_of_sight[ray_idx]
            distance = ray['distance']
            obj_type = ray['type']
            
            # Skip irrelevant types (unknown=0, people=5, explorable_area=7)
            if obj_type in [0, 5, 7]:
                continue
                
            # Track closest object
            if distance > 0 and distance < min_distance:
                min_distance = distance
                closest_distance = discretize_distance(distance)
                closest_object_type = obj_type
            
            # Set object presence flags
            if obj_type == ObjectType.OBSTACLE:
                has_obstacle = True
            elif obj_type == ObjectType.CHECKPOINT:
                has_checkpoint = True
            elif obj_type == ObjectType.TRAP:
                has_trap = True
            elif obj_type == ObjectType.GOAL:
                has_goal = True
            elif obj_type == ObjectType.FOOD:
                has_food = True
        
        direction_info.append({
            'direction': direction,
            'has_obstacle': has_obstacle,
            'has_checkpoint': has_checkpoint,
            'has_trap': has_trap,
            'has_goal': has_goal,
            'has_food': has_food,
            'closest_distance': int(closest_distance),
            'closest_object_type': closest_object_type
        })
    
    return direction_info

def state_to_discrete_state(state: State) -> DiscreteState:
    """Convert full state to discrete state for MCTS"""
    # Simplify agent status: only care about normal(0) vs fell_down(1)
    # Ignore won(2)/dead(3) as they should be terminal states
    agent_status = 1 if state['state'] == 1 else 0  # fell_down or normal
    
    return {
        'agent_status': agent_status,
        'hunger_level': discretize_hunger(state['hunger']),
        'rays': aggregate_rays_to_8_directions(state['line_of_sight']),
        'timestamp': state['timestamp']
    }

def discrete_state_to_hash(discrete_state: DiscreteState) -> str:
    """
    Convert discrete state to hash string for MCTS node identification
    Format: status_hunger_ray1_ray2_..._ray8
    Each ray: direction_obstacles_checkpoints_traps_goals_food_distance_type
    """
    parts = [
        str(discrete_state['agent_status']),
        str(discrete_state['hunger_level'])
    ]
    
    for ray in discrete_state['rays']:
        ray_str = f"{ray['direction']}" \
                 f"{'1' if ray['has_obstacle'] else '0'}" \
                 f"{'1' if ray['has_checkpoint'] else '0'}" \
                 f"{'1' if ray['has_trap'] else '0'}" \
                 f"{'1' if ray['has_goal'] else '0'}" \
                 f"{'1' if ray['has_food'] else '0'}" \
                 f"{ray['closest_distance']}" \
                 f"{ray['closest_object_type']}"
        parts.append(ray_str)
    
    return '_'.join(parts)

def calculate_state_space_size() -> dict:
    """Calculate theoretical state space size for MCTS analysis"""
    agent_status_combinations = 2  # normal, fell_down only
    hunger_level_combinations = len(HungerLevel)  # 3 levels
    distance_combinations = len(DistanceCategory)  # 5 distances
    object_type_combinations = 5 + 1  # 5 relevant object types + no object
    
    # Per direction: 5 boolean flags + 1 distance + 1 object type
    per_direction_combinations = (2**5) * distance_combinations * object_type_combinations
    
    # 8 directions
    total_ray_combinations = per_direction_combinations ** 8
    
    # Total state space
    total_combinations = agent_status_combinations * hunger_level_combinations * total_ray_combinations
    
    return {
        'agent_status': agent_status_combinations,
        'hunger_levels': hunger_level_combinations,
        'per_direction': per_direction_combinations,
        'total_ray_combinations': total_ray_combinations,
        'total_state_space': total_combinations,
        'log2_total': total_combinations.bit_length() - 1 if total_combinations > 0 else 0
    }

# ============================================================================
# OPTIMIZED CONVERSION FUNCTIONS
# ============================================================================

def discretize_distance_optimized(distance: float) -> int:
    """Convert distance to optimized 3-category system"""
    if distance <= 10:
        return OptimizedDistanceCategory.CLOSE
    elif distance <= 25:
        return OptimizedDistanceCategory.MEDIUM
    else:
        return OptimizedDistanceCategory.FAR

def classify_important_object(object_type: int) -> int:
    """Convert object type to consolidated importance categories"""
    if object_type == 1:  # OBSTACLE
        return ImportantObjectType.OBSTACLE
    elif object_type in [2, 4]:  # CHECKPOINT or GOAL
        return ImportantObjectType.PROGRESS
    elif object_type == 3:  # TRAP
        return ImportantObjectType.DANGER  
    elif object_type == 6:  # FOOD
        return ImportantObjectType.RESOURCE
    else:
        return ImportantObjectType.NONE

def calculate_spatial_context(location: Location, snapshot: Snapshot) -> SpatialContext:
    """
    Calculate coarse spatial context for position awareness
    Maps world coordinates to 8x8 grid for approximate positioning
    """
    # Map world coordinates to coarse grid (8x8)
    world_size = 100  # Assume world is roughly 100x100 units
    grid_size = 8
    
    grid_x = min(max(0, int((location["x"] + world_size/2) * grid_size / world_size)), grid_size - 1)
    grid_z = min(max(0, int((location["z"] + world_size/2) * grid_size / world_size)), grid_size - 1)
    
    # Simple facing direction (could be enhanced with actual agent orientation)
    facing = 0  # Default to forward, could be improved with movement history
    
    return {
        "grid_x": grid_x,
        "grid_z": grid_z, 
        "facing": facing
    }

def aggregate_rays_to_4_zones_optimized(line_of_sight: List[LineOfSight]) -> List[OptimizedRayInfo]:
    """
    Aggregate rays into 4 cardinal zones instead of 8 directions
    Each zone covers 90° arc for simpler spatial reasoning
    """
    num_rays = len(line_of_sight)
    if num_rays == 0:
        return []
    
    # Initialize 4 zones
    zone_info = []
    rays_per_zone = max(1, num_rays // 4)
    
    for zone in range(4):
        # Calculate which rays belong to this zone  
        start_ray = zone * rays_per_zone
        end_ray = min(start_ray + rays_per_zone, num_rays)
        
        # Aggregate rays in this zone
        has_obstacle = False
        most_important_object = ImportantObjectType.NONE
        closest_distance = OptimizedDistanceCategory.FAR
        min_distance = float('inf')
        
        for ray_idx in range(start_ray, end_ray):
            if ray_idx >= len(line_of_sight):
                break
                
            ray = line_of_sight[ray_idx]
            distance = ray['distance']
            obj_type = ray['type']
            
            # Skip irrelevant types (unknown=0, people=5, explorable_area=7)
            if obj_type in [0, 5, 7]:
                continue
            
            # Track closest object for distance
            if distance > 0 and distance < min_distance:
                min_distance = distance
                closest_distance = discretize_distance_optimized(distance)
            
            # Set obstacle flag
            if obj_type == 1:  # OBSTACLE
                has_obstacle = True
            
            # Track most important object (priority: DANGER > PROGRESS > RESOURCE > OBSTACLE)
            important_obj = classify_important_object(obj_type)
            if important_obj != ImportantObjectType.NONE:
                # Priority ordering for "most important"
                if (most_important_object == ImportantObjectType.NONE or
                    (important_obj == ImportantObjectType.DANGER) or
                    (important_obj == ImportantObjectType.PROGRESS and most_important_object not in [ImportantObjectType.DANGER]) or
                    (important_obj == ImportantObjectType.RESOURCE and most_important_object == ImportantObjectType.OBSTACLE)):
                    most_important_object = important_obj
        
        zone_info.append({
            'zone': zone,
            'has_obstacle': has_obstacle,
            'important_object': int(most_important_object),
            'closest_distance': int(closest_distance)
        })
    
    return zone_info

def state_to_optimized_discrete_state(state: State) -> OptimizedDiscreteState:
    """Convert full state to optimized discrete state for efficient learning"""
    # Agent status (same as before)
    agent_status = 1 if state['state'] == 1 else 0  # fell_down or normal
    
    # Spatial context for position awareness
    spatial_context = calculate_spatial_context(state['location'], state['snapshot'])
    
    return {
        'agent_status': agent_status,
        'hunger_level': discretize_hunger(state['hunger']),
        'spatial_context': spatial_context,
        'zones': aggregate_rays_to_4_zones_optimized(state['line_of_sight']),
        'timestamp': state['timestamp']
    }

def optimized_discrete_state_to_hash(discrete_state: OptimizedDiscreteState) -> str:
    """
    Convert optimized discrete state to hash for MCTS node identification
    Much shorter hash due to reduced complexity
    """
    parts = [
        str(discrete_state['agent_status']),
        str(discrete_state['hunger_level']),
        f"{discrete_state['spatial_context']['grid_x']}{discrete_state['spatial_context']['grid_z']}{discrete_state['spatial_context']['facing']}"
    ]
    
    for zone in discrete_state['zones']:
        zone_str = f"{zone['zone']}" \
                  f"{'1' if zone['has_obstacle'] else '0'}" \
                  f"{zone['important_object']}" \
                  f"{zone['closest_distance']}"
        parts.append(zone_str)
    
    return '_'.join(parts)

def calculate_optimized_state_space_size() -> dict:
    """Calculate optimized state space size - much smaller than original"""
    agent_status_combinations = 2  # normal, fell_down
    hunger_level_combinations = 3  # low, medium, high
    
    # Spatial context: 8×8 grid + 4 facing directions = 8×8×4 = 256
    spatial_combinations = 8 * 8 * 4
    
    # Per zone: obstacle(2) × important_object(5) × distance(3) = 30
    per_zone_combinations = 2 * 5 * 3
    
    # 4 zones
    total_zone_combinations = per_zone_combinations ** 4
    
    # Total state space
    total_combinations = (agent_status_combinations * 
                         hunger_level_combinations * 
                         spatial_combinations * 
                         total_zone_combinations)
    
    return {
        'agent_status': agent_status_combinations,
        'hunger_levels': hunger_level_combinations,
        'spatial_context': spatial_combinations,
        'per_zone': per_zone_combinations,
        'total_zone_combinations': total_zone_combinations,
        'total_state_space': total_combinations,
        'log2_total': total_combinations.bit_length() - 1 if total_combinations > 0 else 0,
        'reduction_factor': int(4.0e24 / total_combinations) if total_combinations > 0 else 0
    }

# ============================================================================
# BACKWARD COMPATIBILITY AND MIGRATION
# ============================================================================

def migrate_to_optimized_state(old_discrete_state: DiscreteState) -> OptimizedDiscreteState:
    """
    Migrate from old 66-feature state to new 20-feature optimized state
    Useful for transitioning existing saved states/models
    """
    # Convert old rays (8 directions) to new zones (4 directions)
    old_rays = old_discrete_state.get('rays', [])
    
    # Group old rays into 4 zones (2 old directions per new zone)
    new_zones = []
    for zone_idx in range(4):
        # Map 8 old directions to 4 new zones
        old_dir_indices = [(zone_idx * 2) % 8, ((zone_idx * 2) + 1) % 8]
        
        has_obstacle = False
        most_important = ImportantObjectType.NONE
        closest_dist = OptimizedDistanceCategory.FAR
        
        for old_idx in old_dir_indices:
            if old_idx < len(old_rays):
                old_ray = old_rays[old_idx]
                
                if old_ray.get('has_obstacle', False):
                    has_obstacle = True
                
                # Convert old object detection to new importance system
                if old_ray.get('has_checkpoint', False) or old_ray.get('has_goal', False):
                    most_important = ImportantObjectType.PROGRESS
                elif old_ray.get('has_trap', False):
                    most_important = ImportantObjectType.DANGER
                elif old_ray.get('has_food', False) and most_important == ImportantObjectType.NONE:
                    most_important = ImportantObjectType.RESOURCE
                
                # Convert distance (map old 5-category to new 3-category)
                old_dist = old_ray.get('closest_distance', 4)
                if old_dist <= 1:  # VERY_CLOSE or CLOSE
                    closest_dist = OptimizedDistanceCategory.CLOSE
                elif old_dist <= 2:  # MEDIUM
                    closest_dist = OptimizedDistanceCategory.MEDIUM
                else:
                    closest_dist = OptimizedDistanceCategory.FAR
        
        new_zones.append({
            'zone': zone_idx,
            'has_obstacle': has_obstacle,
            'important_object': int(most_important),
            'closest_distance': int(closest_dist)
        })
    
    # Create dummy spatial context (would need actual position data for real migration)
    spatial_context = {
        'grid_x': 4,  # Default to center
        'grid_z': 4,  # Default to center
        'facing': 0   # Default to forward
    }
    
    return {
        'agent_status': old_discrete_state.get('agent_status', 0),
        'hunger_level': old_discrete_state.get('hunger_level', 0),
        'spatial_context': spatial_context,
        'zones': new_zones,
        'timestamp': old_discrete_state.get('timestamp', 0)
    }

# Legacy functions for backward compatibility (keep old interface)
def state_to_discrete_state(state: State) -> DiscreteState:
    """Legacy function - converts to old format for backward compatibility"""
    # Implementation stays the same as before
    agent_status = 1 if state['state'] == 1 else 0
    return {
        'agent_status': agent_status,
        'hunger_level': discretize_hunger(state['hunger']),
        'rays': aggregate_rays_to_8_directions(state['line_of_sight']),
        'timestamp': state['timestamp']
    }

# New optimized functions
def state_to_optimized_state(state: State) -> OptimizedDiscreteState:
    """New optimized conversion function"""
    return state_to_optimized_discrete_state(state)

