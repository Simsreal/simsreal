import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Dict, Any
from loguru import logger
from simsreal_types import State, LineOfSight, Snapshot, DiscreteState, ActionType
import time
import copy

# Add this import at the top
try:
    from animation_generator import AnimationGenerator
    MANIM_AVAILABLE = True
except ImportError:
    MANIM_AVAILABLE = False
    logger.warning("Manim not available - animations will be disabled")


class SnapshotManager:
    def __init__(self, base_dir: str = "snapshots", enable_animations: bool = False, debug_frames: bool = False, disable_images: bool = False):
        self.base_dir = base_dir
        self.current_run = 0
        self.current_run_dir = ""
        self.frame_count = 0
        self.last_state = 0
        self.enable_animations = enable_animations and MANIM_AVAILABLE
        self.debug_frames = debug_frames
        self.disable_images = disable_images  # New flag to skip PNG generation
        
        # Exploration tracking
        self.explorations_dir = ""
        self.current_exploration_count = 0
        
        # Batch logging for exploration sessions
        self.exploration_batch = {
            'last_summary_time': time.time(),
            'batch_count': 0,
            'termination_counts': {},
            'total_steps': 0,
            'total_reward': 0.0,
            'batch_interval': 10.0  # Log summary every 10 seconds instead of individual logs
        }
        
        # Snapshot generation parameters
        self.snapshot_size = 100  # 100x100 grid
        self.snapshot_resolution = 1.0  # 1 meter per pixel
        
        if self.enable_animations:
            self.animation_generator = AnimationGenerator(base_dir)
            logger.info("Animation generation enabled - will generate on-demand")
        else:
            logger.info("Animation generation disabled - use generate_animation_for_current_run() to create animations on-demand")
        
        if self.debug_frames:
            logger.info("Debug frame saving enabled - frames will be saved for analysis")
        
        if self.disable_images:
            logger.info("Image generation DISABLED - only JSON metadata and exploration tracking will be saved")
        
        self._initialize_run_directory()
    
    def _initialize_run_directory(self):
        """Initialize the run directory based on existing runs"""
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Find the highest existing run number
        existing_runs = []
        for item in os.listdir(self.base_dir):
            if item.startswith("run") and os.path.isdir(os.path.join(self.base_dir, item)):
                try:
                    run_num = int(item[3:])  # Extract number after "run"
                    existing_runs.append(run_num)
                except ValueError:
                    continue
        
        if existing_runs:
            self.current_run = max(existing_runs) + 1
        else:
            self.current_run = 0
        
        self._create_run_directory()
    
    def _create_run_directory(self):
        """Create directory for current run"""
        self.current_run_dir = os.path.join(self.base_dir, f"run{self.current_run}")
        os.makedirs(self.current_run_dir, exist_ok=True)
        
        # Create explorations subdirectory
        self.explorations_dir = os.path.join(self.current_run_dir, "explorations")
        os.makedirs(self.explorations_dir, exist_ok=True)
        
        self.frame_count = 0
        self.current_exploration_count = 0
        logger.info(f"Created new run directory: {self.current_run_dir}")
        logger.info(f"Created explorations directory: {self.explorations_dir}")
    
    def _check_episode_end(self, state: int) -> bool:
        """Check if episode ended (state 2=won or 3=dead)"""
        return state in [2, 3]

    # ============================================================================
    # EXPLORATION TRACKING METHODS
    # ============================================================================
    
    def create_exploration_session(self, mcts_iteration: int, root_state: State) -> str:
        """
        Create a new exploration session directory and return the session ID
        
        Args:
            mcts_iteration: Current MCTS iteration number
            root_state: The root state from which exploration starts
            
        Returns:
            exploration_id: Unique identifier for this exploration session
        """
        exploration_id = f"{self.current_exploration_count:04d}"
        exploration_dir = os.path.join(self.explorations_dir, exploration_id)
        os.makedirs(exploration_dir, exist_ok=True)
        
        # Save root state information
        root_info = {
            "exploration_id": exploration_id,
            "mcts_iteration": mcts_iteration,
            "timestamp": int(time.time() * 1000),
            "root_state": {
                "location": root_state["location"],
                "agent_status": root_state["state"],
                "hunger": root_state["hunger"],
                "snapshot_size": f"{root_state['snapshot']['width']}x{root_state['snapshot']['height']}",
                "num_rays": len(root_state["line_of_sight"])
            }
        }
        
        with open(os.path.join(exploration_dir, "exploration_info.json"), 'w') as f:
            json.dump(root_info, f, indent=2)
        
        # Save root state snapshot as baseline
        self._save_exploration_snapshot(exploration_dir, "root_state", root_state, {
            "step": 0,
            "action": "initial",
            "is_root": True
        })
        
        self.current_exploration_count += 1
        # logger.debug(f"Created exploration session {exploration_id} in {exploration_dir}")
        
        return exploration_id
    
    def save_exploration_step(self, exploration_id: str, step: int, state: State, 
                            action: ActionType, reward: float, is_terminal: bool,
                            additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a single step in an exploration rollout
        
        Args:
            exploration_id: ID of the exploration session
            step: Step number in the rollout
            state: Current state after taking action
            action: Action that was taken
            reward: Reward received for this step
            is_terminal: Whether this step ended the rollout
            additional_info: Additional debugging information
        """
        exploration_dir = os.path.join(self.explorations_dir, exploration_id)
        
        if not os.path.exists(exploration_dir):
            logger.warning(f"Exploration directory {exploration_dir} does not exist")
            return
        
        # Prepare step metadata
        step_info = {
            "step": step,
            "action": action.name if hasattr(action, 'name') else str(action),
            "reward": reward,
            "is_terminal": is_terminal,
            "location": state["location"],
            "agent_status": state["state"],
            "hunger": state["hunger"],
            "timestamp": int(time.time() * 1000)
        }
        
        if additional_info:
            step_info.update(additional_info)
        
        # Save step snapshot with step info
        snapshot_name = f"step_{step:03d}"
        if is_terminal:
            snapshot_name += "_terminal"
            
        self._save_exploration_snapshot(exploration_dir, snapshot_name, state, step_info)
    
    def finalize_exploration_session(self, exploration_id: str, total_reward: float, 
                                   total_steps: int, termination_reason: str,
                                   rollout_summary: Optional[Dict[str, Any]] = None) -> None:
        """
        Finalize an exploration session with summary information
        
        Args:
            exploration_id: ID of the exploration session
            total_reward: Total reward accumulated during rollout
            total_steps: Total number of steps taken
            termination_reason: Why the rollout ended (goal, hunger, stagnation, max_steps)
            rollout_summary: Additional summary information
        """
        exploration_dir = os.path.join(self.explorations_dir, exploration_id)
        
        if not os.path.exists(exploration_dir):
            logger.warning(f"Exploration directory {exploration_dir} does not exist")
            return
        
        # Create rollout summary
        summary = {
            "exploration_id": exploration_id,
            "total_reward": total_reward,
            "total_steps": total_steps,
            "termination_reason": termination_reason,
            "avg_reward_per_step": total_reward / max(1, total_steps),
            "completed_timestamp": int(time.time() * 1000)
        }
        
        if rollout_summary:
            summary.update(rollout_summary)
        
        # Save summary
        with open(os.path.join(exploration_dir, "rollout_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Update batch tracking instead of individual debug log
        self._update_exploration_batch(exploration_id, total_reward, total_steps, termination_reason)
    
    def _update_exploration_batch(self, exploration_id: str, reward: float, steps: int, reason: str):
        """Update batch tracking and log summary when interval reached"""
        batch = self.exploration_batch
        batch['batch_count'] += 1
        batch['total_reward'] += reward
        batch['total_steps'] += steps
        batch['termination_counts'][reason] = batch['termination_counts'].get(reason, 0) + 1
        
        current_time = time.time()
        if current_time - batch['last_summary_time'] >= batch['batch_interval']:
            # Log batch summary
            avg_reward = batch['total_reward'] / max(1, batch['batch_count'])
            avg_steps = batch['total_steps'] / max(1, batch['batch_count'])
            
            # Format termination reasons summary
            reason_summary = []
            for reason, count in sorted(batch['termination_counts'].items()):
                percentage = (count / batch['batch_count']) * 100
                reason_summary.append(f"{reason}:{count}({percentage:.0f}%)")
            
            logger.info(f"Exploration batch summary: {batch['batch_count']} sessions completed "
                       f"- avg_reward={avg_reward:.3f}, avg_steps={avg_steps:.1f}, "
                       f"reasons=[{', '.join(reason_summary)}]")
            
            # Reset batch tracking
            batch['last_summary_time'] = current_time
            batch['batch_count'] = 0
            batch['termination_counts'] = {}
            batch['total_steps'] = 0
            batch['total_reward'] = 0.0

    def _save_exploration_snapshot(self, exploration_dir: str, snapshot_name: str, 
                                 state: State, step_info: Dict[str, Any]) -> None:
        """
        Save a snapshot for exploration with step information overlay
        
        Args:
            exploration_dir: Directory for this exploration session
            snapshot_name: Name for this snapshot
            state: Current state
            step_info: Information about this step
        """
        try:
            # Always save step metadata as JSON
            metadata_filepath = os.path.join(exploration_dir, f"{snapshot_name}.json")
            step_metadata = {
                "state": {
                    "location": state["location"],
                    "agent_status": state["state"],
                    "hunger": state["hunger"],
                    "hitpoint": state["hitpoint"],
                    "timestamp": state["timestamp"]
                },
                "step_info": step_info,
                "snapshot_info": {
                    "width": state.get('snapshot', {}).get('width', 0),
                    "height": state.get('snapshot', {}).get('height', 0),
                    "resolution": state.get('snapshot', {}).get('resolution', 0)
                }
            }
            
            with open(metadata_filepath, 'w') as f:
                json.dump(step_metadata, f, indent=2)
            
            # Skip image generation if disabled
            if self.disable_images:
                return
            
            # Generate the base raycast image
            line_of_sight = state.get('line_of_sight', [])
            image_array = self._raycast_to_image(line_of_sight)
            
            # Create PIL image
            image = Image.fromarray(image_array, mode='RGB')
            
            # Add exploration-specific information overlay
            image = self._add_exploration_overlay(image, state, step_info)
            
            # Save the image
            filepath = os.path.join(exploration_dir, f"{snapshot_name}.png")
            image.save(filepath)
                
        except Exception as e:
            logger.error(f"Error saving exploration snapshot {snapshot_name}: {e}")
    
    def _add_exploration_overlay(self, image: Image.Image, state: State, step_info: Dict[str, Any]) -> Image.Image:
        """Add exploration-specific information overlay to the image"""
        # Create a larger image to accommodate exploration info
        img_width, img_height = image.size
        info_width = 250
        new_width = img_width + info_width
        
        # Create new image with dark background
        combined = Image.new('RGB', (new_width, img_height), (20, 20, 20))
        combined.paste(image, (0, 0))
        
        # Draw exploration info
        draw = ImageDraw.Draw(combined)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Exploration info items
        info_items = [
            ("EXPLORATION DATA", (255, 255, 0)),
            ("", (0, 0, 0)),  # Spacer
            (f"Step: {step_info.get('step', 'N/A')}", (255, 255, 255)),
            (f"Action: {step_info.get('action', 'N/A')}", (255, 255, 255)),
            (f"Reward: {step_info.get('reward', 0):.3f}", (0, 255, 0) if step_info.get('reward', 0) >= 0 else (255, 100, 100)),
            (f"Terminal: {'Yes' if step_info.get('is_terminal', False) else 'No'}", 
             (255, 100, 100) if step_info.get('is_terminal', False) else (200, 200, 200)),
            ("", (0, 0, 0)),  # Spacer
            ("AGENT STATE", (255, 255, 0)),
            ("", (0, 0, 0)),  # Spacer
            (f"Pos: ({state['location']['x']}, {state['location']['z']})", (200, 200, 200)),
            (f"Status: {['Normal', 'Fell', 'Won', 'Dead'][state['state']]}", (200, 200, 200)),
            (f"Hunger: {state['hunger']:.1f}", (200, 200, 200)),
            (f"HP: {state['hitpoint']}", (200, 200, 200)),
            ("", (0, 0, 0)),  # Spacer
            ("SNAPSHOT INFO", (255, 255, 0)),
            ("", (0, 0, 0)),  # Spacer
            (f"Size: {state.get('snapshot', {}).get('width', 0)}x{state.get('snapshot', {}).get('height', 0)}", (200, 200, 200)),
            (f"Res: {state.get('snapshot', {}).get('resolution', 0):.1f}m/px", (200, 200, 200)),
            (f"Rays: {len(state.get('line_of_sight', []))}", (200, 200, 200)),
        ]
        
        # Add additional info if provided
        if 'intrinsic_rewards' in step_info:
            info_items.extend([
                ("", (0, 0, 0)),  # Spacer
                ("INTRINSIC REWARDS", (255, 255, 0)),
                ("", (0, 0, 0)),  # Spacer
            ])
            for reward_type, value in step_info['intrinsic_rewards'].items():
                color = (0, 255, 0) if value >= 0 else (255, 100, 100)
                info_items.append((f"{reward_type}: {value:.3f}", color))
        
        # Draw all info items
        y_offset = 20
        for text, color in info_items:
            if text:  # Skip empty spacer lines
                draw.text((img_width + 10, y_offset), text, fill=color, font=font)
            y_offset += 16
        
        return combined
    
    def get_exploration_summary(self, run_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Get summary of all explorations for a specific run
        
        Args:
            run_number: Run number to analyze (None for current run)
            
        Returns:
            Dictionary with exploration statistics
        """
        if run_number is None:
            explorations_path = self.explorations_dir
            run_number = self.current_run
        else:
            explorations_path = os.path.join(self.base_dir, f"run{run_number}", "explorations")
        
        if not os.path.exists(explorations_path):
            return {"error": f"No explorations found for run {run_number}"}
        
        # Collect all exploration summaries
        exploration_dirs = [d for d in os.listdir(explorations_path) 
                          if os.path.isdir(os.path.join(explorations_path, d))]
        
        summaries = []
        total_reward = 0
        total_steps = 0
        termination_reasons = {}
        
        for exp_dir in sorted(exploration_dirs):
            summary_file = os.path.join(explorations_path, exp_dir, "rollout_summary.json")
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    summaries.append(summary)
                    total_reward += summary.get('total_reward', 0)
                    total_steps += summary.get('total_steps', 0)
                    
                    reason = summary.get('termination_reason', 'unknown')
                    termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
        
        return {
            "run_number": run_number,
            "total_explorations": len(summaries),
            "total_reward": total_reward,
            "total_steps": total_steps,
            "avg_reward_per_exploration": total_reward / max(1, len(summaries)),
            "avg_steps_per_exploration": total_steps / max(1, len(summaries)),
            "termination_reasons": termination_reasons,
            "explorations": summaries
        }

    # ============================================================================
    # EXISTING METHODS (unchanged)
    # ============================================================================
    
    def generate_snapshot_from_raycast(self, agent_location: dict, line_of_sight: List[LineOfSight], 
                                     max_distance: float = 100.0) -> Snapshot:
        """Generate a 2D snapshot from raycast data"""
        # Initialize snapshot grid
        snapshot_data = [[0 for _ in range(self.snapshot_size)] for _ in range(self.snapshot_size)]
        
        # Calculate agent position in snapshot coordinates
        agent_x = agent_location['x']
        agent_z = agent_location['z']
        
        # Origin is at bottom-left of the snapshot, agent at center
        origin_x = agent_x - (self.snapshot_size // 2) * self.snapshot_resolution
        origin_z = agent_z - (self.snapshot_size // 2) * self.snapshot_resolution
        
        # Convert agent world position to snapshot grid coordinates
        agent_grid_x = self.snapshot_size // 2
        agent_grid_z = self.snapshot_size // 2
        
        num_rays = len(line_of_sight)
        
        for i, ray in enumerate(line_of_sight):
            # Calculate angle for this ray
            angle = (i / num_rays) * 2 * np.pi - np.pi / 2
            
            # Get ray distance and type
            distance = ray.get('distance', max_distance)
            if distance <= 0:
                distance = max_distance
            
            ray_type = ray.get('type', 0)
            
            # Calculate hit position in world coordinates
            hit_x = agent_x - distance * np.cos(angle)  # Unity coordinate system
            hit_z = agent_z - distance * np.sin(angle)
            
            # Convert to snapshot grid coordinates
            grid_x = int((hit_x - origin_x) / self.snapshot_resolution)
            grid_z = int((hit_z - origin_z) / self.snapshot_resolution)
            
            # Ensure coordinates are within bounds
            if 0 <= grid_x < self.snapshot_size and 0 <= grid_z < self.snapshot_size:
                # Mark object at hit location (if any)
                if ray_type > 0:
                    snapshot_data[grid_z][grid_x] = ray_type
                
                # Mark explorable area along the ray path
                self._mark_ray_path_in_snapshot(snapshot_data, agent_grid_x, agent_grid_z, 
                                              grid_x, grid_z, max_distance, distance)
        
        return {
            "data": snapshot_data,
            "width": self.snapshot_size,
            "height": self.snapshot_size,
            "resolution": self.snapshot_resolution,
            "origin": {"x": int(origin_x), "z": int(origin_z)},
            "timestamp": int(time.time() * 1000)
        }
    
    def _mark_ray_path_in_snapshot(self, snapshot_data: List[List[int]], 
                                  start_x: int, start_z: int, end_x: int, end_z: int,
                                  max_distance: float, actual_distance: float):
        """Mark the ray path as explorable area in the snapshot"""
        # Use Bresenham's line algorithm to mark the ray path
        points = self._bresenham_line(start_x, start_z, end_x, end_z)
        
        for x, z in points:
            if 0 <= x < self.snapshot_size and 0 <= z < self.snapshot_size:
                # Only mark as explorable if no object is already there
                if snapshot_data[z][x] == 0:
                    snapshot_data[z][x] = 7  # 7 = explorable_area
    
    def _bresenham_line(self, x0: int, z0: int, x1: int, z1: int) -> List[tuple]:
        """Bresenham's line algorithm for marking paths"""
        points = []
        dx = abs(x1 - x0)
        dz = abs(z1 - z0)
        x, z = x0, z0
        x_inc = 1 if x1 > x0 else -1
        z_inc = 1 if z1 > z0 else -1
        error = dx - dz

        while True:
            points.append((x, z))
            if x == x1 and z == z1:
                break

            e2 = 2 * error
            if e2 > -dz:
                error -= dz
                x += x_inc
            if e2 < dx:
                error += dx
                z += z_inc

        return points
    
    def _raycast_to_image(self, line_of_sight: List[LineOfSight], max_distance: float = 100.0) -> np.ndarray:
        """
        Convert ray-cast data to visible image with clear object representation and explorable area
        Forward movement (increasing Z) goes upward in the image
        Left/right matches Unity coordinate system (left = -X, right = +X)
        """
        img_size = 400
        image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        if not line_of_sight:
            return image
        
        # Position agent near bottom of image (forward is up)
        center_x = img_size // 2
        center_y = int(img_size * 0.8)
        
        # Number of rays
        num_rays = len(line_of_sight)
        
        # Define distinct colors for each object type
        type_colors = {
            0: (128, 128, 128),  # unknown - gray
            1: (255, 0, 0),      # obstacle - red
            2: (0, 255, 0),      # checkpoint - green  
            3: (255, 165, 0),    # trap - orange
            4: (0, 0, 255),      # goal - blue
            5: (255, 255, 0),    # people - yellow
            6: (255, 0, 255),    # food - magenta
            7: (0, 140, 0)       # explorable_area - dark green
        }
        
        # Calculate explorable area - ALL rays define the boundary
        ray_endpoints = []
        
        max_ray_length = min(center_y - 20, img_size - center_x - 20, center_x - 20)
        
        for i, ray in enumerate(line_of_sight):
            # Calculate angle for this ray
            angle = (i / num_rays) * 2 * np.pi - np.pi / 2
            
            # For explorable area: use actual ray distance OR max distance if no hit
            distance = ray.get('distance', max_distance)
            if distance <= 0:  # No hit detected, use max range
                distance = max_distance
            
            # Normalize distance
            normalized_distance = min(distance / max_distance, 1.0)
            ray_length = normalized_distance * max_ray_length
            
            # Calculate position
            end_x = int(center_x - ray_length * np.cos(angle))
            end_y = int(center_y - ray_length * np.sin(angle))
            
            # Ensure coordinates are within bounds
            end_x = max(0, min(img_size - 1, end_x))
            end_y = max(0, min(img_size - 1, end_y))
            
            ray_endpoints.append((end_x, end_y, angle, distance, ray.get('type', 0)))
        
        # Draw explorable area FIRST (so it appears behind everything else)
        self._draw_explorable_area_solid(image, center_x, center_y, ray_endpoints)
        
        # Draw objects on top (no individual ray lines)
        for i, ray in enumerate(line_of_sight):
            angle = (i / num_rays) * 2 * np.pi - np.pi / 2
            
            # Use actual distance for drawing objects
            distance = ray.get('distance', max_distance)
            if distance <= 0:
                continue  # Skip if no object detected
                
            normalized_distance = min(distance / max_distance, 1.0)
            ray_length = normalized_distance * max_ray_length
            
            end_x = int(center_x - ray_length * np.cos(angle))
            end_y = int(center_y - ray_length * np.sin(angle))
            end_x = max(0, min(img_size - 1, end_x))
            end_y = max(0, min(img_size - 1, end_y))
            
            object_type = ray.get('type', 0)
            
            # Only draw circles for actual detected objects
            if object_type > 0:
                color = type_colors.get(object_type, (128, 128, 128))
                circle_radius = max(4, int(10 - normalized_distance * 5))  # Slightly larger objects
                self._draw_circle(image, end_x, end_y, circle_radius, color)
        
        # Draw center point (agent position) - larger and white
        self._draw_circle(image, center_x, center_y, 8, (255, 255, 255))
        
        # Draw direction indicator (small arrow pointing forward/up)
        arrow_length = 18
        arrow_tip_x = center_x
        arrow_tip_y = center_y - arrow_length
        arrow_base_x = center_x
        arrow_base_y = center_y - 10
        
        # Draw arrow shaft
        self._draw_line(image, arrow_base_x, arrow_base_y, arrow_tip_x, arrow_tip_y, (255, 255, 255))
        # Draw arrow head
        self._draw_line(image, arrow_tip_x, arrow_tip_y, arrow_tip_x - 4, arrow_tip_y + 4, (255, 255, 255))
        self._draw_line(image, arrow_tip_x, arrow_tip_y, arrow_tip_x + 4, arrow_tip_y + 4, (255, 255, 255))
        
        return image
    
    def _draw_explorable_area_solid(self, image: np.ndarray, center_x: int, center_y: int, ray_endpoints: List[tuple]):
        """Draw the explorable area as a solid filled region"""
        if len(ray_endpoints) < 3:
            return
        
        # Sort endpoints by angle to create proper polygon
        ray_endpoints.sort(key=lambda p: p[2])  # Sort by angle
        
        # Create polygon points for the explorable area
        polygon_points = []
        
        # Add all ray endpoints to form the boundary
        for endpoint in ray_endpoints:
            polygon_points.append((endpoint[0], endpoint[1]))
        
        # Fill the entire explorable area with solid green
        if len(polygon_points) >= 3:
            self._draw_filled_polygon(image, polygon_points, (0, 140, 0), alpha=0.5)
            
            # Draw a subtle border around the explorable area
            for i in range(len(polygon_points)):
                curr_point = polygon_points[i]
                next_point = polygon_points[(i + 1) % len(polygon_points)]
                self._draw_line(image, curr_point[0], curr_point[1], 
                               next_point[0], next_point[1], (0, 200, 0))
    
    def _draw_filled_polygon(self, image: np.ndarray, points: List[tuple], color: tuple, alpha: float = 1.0):
        """Draw a completely filled polygon using improved scan line algorithm"""
        if len(points) < 3:
            return
        
        # Convert to integers and ensure bounds
        points = [(max(0, min(image.shape[1]-1, int(p[0]))), 
                  max(0, min(image.shape[0]-1, int(p[1])))) for p in points]
        
        # Get bounding box
        min_y = max(0, min(p[1] for p in points))
        max_y = min(image.shape[0] - 1, max(p[1] for p in points))
        
        # For each scanline
        for y in range(min_y, max_y + 1):
            intersections = []
            
            # Find all intersections with polygon edges
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                x1, y1 = p1
                x2, y2 = p2
                
                # Check if this edge crosses the scanline
                if y1 != y2:  # Skip horizontal edges
                    # Check if scanline intersects this edge
                    if (y1 <= y < y2) or (y2 <= y < y1):
                        # Calculate intersection point
                        x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                        x_intersect = max(0, min(image.shape[1] - 1, int(x_intersect)))
                        intersections.append(x_intersect)
            
            # Sort intersections
            intersections.sort()
            
            # Fill between pairs of intersections
            for i in range(0, len(intersections) - 1, 2):
                if i + 1 < len(intersections):
                    x_start = int(intersections[i])
                    x_end = int(intersections[i + 1])
                    
                    # Fill the entire span
                    for x in range(x_start, x_end + 1):
                        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                            # Blend colors with alpha
                            original = image[y, x].astype(float)
                            new_color = np.array(color, dtype=float)
                            image[y, x] = ((1 - alpha) * original + alpha * new_color).astype(np.uint8)
    
    def _draw_line(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple):
        """Draw a line on the image"""
        # Simple line drawing using Bresenham's algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        if dx > dy:
            err = dx / 2.0
            while x != x2:
                if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                    image[y, x] = color
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                    image[y, x] = color
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
    
    def _draw_circle(self, image: np.ndarray, cx: int, cy: int, radius: int, color: tuple):
        """Draw a filled circle on the image"""
        for y in range(max(0, cy - radius), min(image.shape[0], cy + radius + 1)):
            for x in range(max(0, cx - radius), min(image.shape[1], cx + radius + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    image[y, x] = color

    def save_snapshot(self, state: State) -> None:
        """Save current state as snapshot image with legend"""
        try:
            # Check if we need to start a new episode
            current_state = state['state']
            
            # Start new run when transitioning from end state (2/3) to any active state (0/1)
            if self._check_episode_end(self.last_state) and current_state in [0, 1]:
                self.current_run += 1
                self._create_run_directory()
                logger.info(f"Episode ended (previous state: {self.last_state}), starting new run {self.current_run}")
                if not self.disable_images:
                    logger.info(f"To generate animation for completed run, use: snapshot_manager.generate_animation_for_run({self.current_run - 1})")
            
            self.last_state = current_state
            
            # Always save state metadata as JSON (only in debug mode or periodically)
            if self.debug_frames or self.frame_count % 50 == 0:
                metadata_filename = f"frame_{self.frame_count:06d}.json"
                metadata_filepath = os.path.join(self.current_run_dir, metadata_filename)
                with open(metadata_filepath, 'w') as f:
                    json.dump(state, f, indent=2)
            
            # Skip image generation if disabled
            if not self.disable_images:
                # Convert ray-cast data to image
                line_of_sight = state.get('line_of_sight', [])
                image_array = self._raycast_to_image(line_of_sight)
                
                # Create PIL image
                image = Image.fromarray(image_array, mode='RGB')
                
                # Add legend and info
                image = self._add_legend_and_info(image, state)
                
                # Save image (always save if debug_frames is enabled, otherwise normal behavior)
                if self.debug_frames or self.frame_count % 10 == 0:
                    filename = f"frame_{self.frame_count:06d}.png"
                    filepath = os.path.join(self.current_run_dir, filename)
                    image.save(filepath)
                    
                    if self.debug_frames:
                        logger.debug(f"Debug frame saved: {filepath}")
            
            self.frame_count += 1
            
            if self.frame_count % 100 == 0:  # Log every 100 frames
                if self.disable_images:
                    logger.debug(f"Processed {self.frame_count} frames (images disabled) in {self.current_run_dir}")
                else:
                    logger.debug(f"Saved {self.frame_count} frames to {self.current_run_dir}")
                
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
    
    def _add_legend_and_info(self, image: Image.Image, state: State) -> Image.Image:
        """Add legend and state information to the image"""
        # Create a larger image to accommodate legend
        img_width, img_height = image.size
        legend_width = 220
        new_width = img_width + legend_width
        
        # Create new image with black background
        combined = Image.new('RGB', (new_width, img_height), (0, 0, 0))
        combined.paste(image, (0, 0))
        
        # Draw legend
        draw = ImageDraw.Draw(combined)
        
        # Try to use a default font, fallback to default if not available
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Legend items
        legend_items = [
            ("Map View:", (255, 255, 255)),
            ("↑ Forward (+Z)", (200, 200, 200)),
            ("← Left (-X) | Right (+X) →", (200, 200, 200)),
            ("Agent: ●→", (255, 255, 255)),
            ("", (0, 0, 0)),  # Spacer
            ("Explorable Area:", (255, 255, 255)),
            ("█ Sensor coverage", (0, 140, 0)),
            ("— Area boundary", (0, 200, 0)),
            ("", (0, 0, 0)),  # Spacer
            ("Objects:", (255, 255, 255)),
            ("○ Unknown", (128, 128, 128)),
            ("○ Obstacle", (255, 0, 0)),
            ("○ Checkpoint", (0, 255, 0)),
            ("○ Trap", (255, 165, 0)),
            ("○ Goal", (0, 0, 255)),
            ("○ People", (255, 255, 0)),
            ("○ Food", (255, 0, 255)),
            ("", (0, 0, 0)),  # Spacer
            ("Agent Info:", (255, 255, 255)),
            (f"Pos: ({state['location']['x']:.1f}, {state['location']['z']:.1f})", (200, 200, 200)),
            (f"HP: {state['hitpoint']}", (200, 200, 200)),
            (f"Hunger: {state['hunger']:.1f}", (200, 200, 200)),
            (f"State: {state['state']}", (200, 200, 200)),
            (f"Frame: {self.frame_count}", (200, 200, 200)),
            ("", (0, 0, 0)),  # Spacer
            ("Snapshot:", (255, 255, 255)),
            (f"Size: {state.get('snapshot', {}).get('width', 0)}x{state.get('snapshot', {}).get('height', 0)}", (200, 200, 200)),
            (f"Res: {state.get('snapshot', {}).get('resolution', 0):.1f}m/px", (200, 200, 200)),
        ]
        
        y_offset = 20
        for text, color in legend_items:
            if text:  # Skip empty spacer lines
                draw.text((img_width + 10, y_offset), text, fill=color, font=font)
            y_offset += 18
        
        return combined

    def get_current_run_info(self) -> dict:
        """Get information about current run"""
        return {
            "run_number": self.current_run,
            "run_directory": self.current_run_dir,
            "frame_count": self.frame_count,
            "explorations_directory": self.explorations_dir,
            "exploration_count": self.current_exploration_count
        } 

    def generate_animation_for_run(self, run_number: int) -> Optional[str]:
        """Generate animation for a specific run on-demand"""
        if not MANIM_AVAILABLE:
            logger.error("Manim not available - cannot generate animations")
            return None
            
        if not hasattr(self, 'animation_generator'):
            self.animation_generator = AnimationGenerator(self.base_dir)
        
        return self.animation_generator.generate_animation_for_run(run_number)
    
    def generate_animation_for_latest_completed_run(self) -> Optional[str]:
        """Generate animation for the most recently completed run"""
        if self.current_run > 0:
            # Generate for previous run (current run - 1 is the completed one)
            return self.generate_animation_for_run(self.current_run - 1)
        else:
            logger.warning("No completed runs to generate animation for")
            return None 