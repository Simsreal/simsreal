import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Callable, Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from pprint import pprint
import threading
import time
import os
from pathlib import Path

from src.utilities.sensor.raycast import (
    process_line_of_sight,
    create_raycast_matrices,
)
from src.utilities.sensor.vision import create_lidar_vision_tensor
from loguru import logger


class MindMap:
    """
    2D occupancy map that tracks environment features based on agent's raycast data.
    Updates continuously and resets when agent state equals 3.
    """
    
    def __init__(self, map_size: int = 512, resolution: float = 0.5, decay_factor: float = 0.99, enable_visualization: bool = True, save_frames: bool = False, output_dir: str = "mindmap_frames"):
        """Initialize the MindMap with reference frame system."""
        self.map_size = map_size
        self.resolution = resolution
        self.decay_factor = decay_factor
        self.enable_visualization = enable_visualization
        self.save_frames = save_frames
        self.base_output_dir = Path(output_dir)
        
        # Episode management
        self.episode_counter = 0
        self.current_episode_dir = None
        self._setup_episode_directory()
        
        # Map channels: [obstacle, enemy, trap, goal, people, food, explored]
        self.num_channels = 7
        self.map_data = np.zeros((self.num_channels, map_size, map_size), dtype=np.float32)
        
        # Type mapping from raycast to channels
        self.type_to_channel = {
            0: -1,  # nil - no update
            1: 0,   # obstacle
            2: 1,   # enemy
            3: 2,   # trap
            4: 3,   # goal
            5: 4,   # people
            6: 5,   # food
        }
        
        # Colors for visualization
        self.colors = {
            0: [0.5, 0.5, 0.5],    # obstacle - gray
            1: [1.0, 0.0, 0.0],    # enemy - red
            2: [1.0, 0.6, 0.0],    # trap - orange
            3: [0.0, 1.0, 0.0],    # goal - green
            4: [1.0, 0.0, 1.0],    # people - magenta
            5: [0.0, 1.0, 1.0],    # food - cyan
            6: [0.25, 0.25, 0.25], # explored - dark gray
        }
        
        # Reference frame system
        self.reference_pos = None  # Will be set on first update or reset
        self.agent_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.map_center = map_size // 2
        
        # Visualization variables
        self.frame_counter = 0
        self.fig = None
        self.ax = None
        self.im = None
        self.agent_marker = None
        self.viz_thread = None
        self.viz_running = False
        self.update_flag = False
        self.viz_lock = threading.Lock()
        
        if self.enable_visualization:
            self._init_matplotlib_visualization()
        
        logger.info(f"MindMap initialized: {map_size}x{map_size}, resolution={resolution}m/px, reference frame system enabled")
    
    def _setup_episode_directory(self):
        """Setup directory for current episode."""
        if self.save_frames:
            # Create base directory if it doesn't exist
            self.base_output_dir.mkdir(exist_ok=True)
            
            # Find the next episode number
            existing_episodes = [d for d in self.base_output_dir.iterdir() 
                               if d.is_dir() and d.name.startswith("episode_")]
            
            if existing_episodes:
                episode_numbers = []
                for ep_dir in existing_episodes:
                    try:
                        ep_num = int(ep_dir.name.split("_")[1])
                        episode_numbers.append(ep_num)
                    except (IndexError, ValueError):
                        continue
                self.episode_counter = max(episode_numbers) + 1 if episode_numbers else 0
            else:
                self.episode_counter = 0
            
            # Create current episode directory
            self.current_episode_dir = self.base_output_dir / f"episode_{self.episode_counter:04d}"
            self.current_episode_dir.mkdir(exist_ok=True)
            
            # Reset frame counter for new episode
            self.frame_counter = 0
            
            logger.info(f"Episode {self.episode_counter}: Saving frames to {self.current_episode_dir}")

    def _init_matplotlib_visualization(self):
        """Initialize matplotlib visualization in a separate thread."""
        try:
            self.viz_running = True
            self.viz_thread = threading.Thread(target=self._visualization_loop, daemon=True)
            self.viz_thread.start()
            logger.info("Matplotlib visualization thread started")
        except Exception as e:
            logger.error(f"Failed to start visualization thread: {e}")
            self.enable_visualization = False
    
    def _visualization_loop(self):
        """Main visualization loop - now saves PNG files instead of displaying."""
        try:
            # Use Agg backend for PNG generation (no GUI required)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Create figure and axis
            self.fig, self.ax = plt.subplots(figsize=(12, 10), dpi=100)
            self.ax.set_title("MindMap Live View")
            self.ax.set_xlabel("Map X (pixels)")
            self.ax.set_ylabel("Map Y (pixels)")
            
            # Create initial empty image
            initial_img = np.zeros((self.map_size, self.map_size, 3))
            self.im = self.ax.imshow(initial_img, origin='lower', extent=[0, self.map_size, 0, self.map_size])
            
            # Add colorbar legend
            self._add_legend()
            
            # Initial agent marker
            self.agent_marker, = self.ax.plot([], [], 'wo', markersize=8, markeredgecolor='black', markeredgewidth=2)
            
            plt.tight_layout()
            
            # Main loop - no GUI display, just PNG saving
            while self.viz_running:
                try:
                    with self.viz_lock:
                        if self.update_flag:
                            self._update_plot()
                            
                            # Save frame as PNG instead of displaying
                            if self.save_frames:
                                self._save_frame()
                            
                            self.update_flag = False
                    
                    # Simple sleep instead of plt.pause()
                    time.sleep(0.1)  # Adjust as needed
                    
                except Exception as e:
                    logger.error(f"Visualization update error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Visualization loop error: {e}")
        finally:
            if self.fig:
                plt.close(self.fig)
            logger.info("Visualization thread ended")
    
    def _add_legend(self):
        """Add a legend to the plot."""
        legend_elements = []
        legend_labels = ["Obstacle", "Enemy", "Trap", "Goal", "People", "Food", "Explored", "Agent"]
        legend_colors = [self.colors[i] for i in range(7)] + [[1.0, 1.0, 1.0]]  # White for agent
        
        for label, color in zip(legend_labels, legend_colors):
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label))
        
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    def _update_plot(self):
        """Update the matplotlib plot with current map data."""
        try:
            # Create RGB image from map data
            viz_img = np.zeros((self.map_size, self.map_size, 3))
            
            # Render each channel with its color (priority order: objects > explored)
            for channel_idx in [6, 0, 1, 2, 3, 4, 5]:  # explored first, then objects
                channel_data = self.map_data[channel_idx]
                mask = channel_data > 0.1
                
                if np.any(mask):
                    color = self.colors[channel_idx]
                    viz_img[mask] = color
            
            # Update image
            self.im.set_data(viz_img)
            
            # Update agent position
            agent_map_x, agent_map_y = self.world_to_map(self.agent_pos["x"], self.agent_pos["y"])
            if self.is_valid_coordinate(agent_map_x, agent_map_y):
                self.agent_marker.set_data([agent_map_x], [agent_map_y])
            
            # Update title with episode info and coordinates
            timestamp = time.strftime("%H:%M:%S")
            if self.reference_pos:
                rel_x = self.agent_pos["x"] - self.reference_pos["x"]
                rel_y = self.agent_pos["y"] - self.reference_pos["y"]
                title = f"Episode {self.episode_counter} - Frame: {self.frame_counter} | Agent: World({self.agent_pos['x']:.1f}, {self.agent_pos['y']:.1f}) Rel({rel_x:.1f}, {rel_y:.1f}) | {timestamp}"
            else:
                title = f"Episode {self.episode_counter} - Frame: {self.frame_counter} | Agent: ({self.agent_pos['x']:.1f}, {self.agent_pos['y']:.1f}) | {timestamp}"
            
            self.ax.set_title(title)
            
        except Exception as e:
            logger.error(f"Error updating plot: {e}")
    
    def world_to_map(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to map indices using reference frame."""
        if self.reference_pos is None:
            # If no reference set, use current position as reference (centered)
            map_x = self.map_center
            map_y = self.map_center
        else:
            # Calculate relative position from reference
            rel_x = world_x - self.reference_pos["x"]
            rel_y = world_y - self.reference_pos["y"]
            
            # Convert to map coordinates (reference pos is at map center)
            map_x = int(self.map_center + rel_x / self.resolution)
            map_y = int(self.map_center + rel_y / self.resolution)
        
        return map_x, map_y
    
    def map_to_world(self, map_x: int, map_y: int) -> Tuple[float, float]:
        """Convert map indices to world coordinates using reference frame."""
        if self.reference_pos is None:
            return 0.0, 0.0
        
        # Calculate relative position from map center
        rel_x = (map_x - self.map_center) * self.resolution
        rel_y = (map_y - self.map_center) * self.resolution
        
        # Add to reference position to get world coordinates
        world_x = self.reference_pos["x"] + rel_x
        world_y = self.reference_pos["y"] + rel_y
        
        return world_x, world_y

    def set_reference_frame(self, agent_position: Dict[str, float]):
        """Set the reference frame based on agent position."""
        self.reference_pos = {
            "x": agent_position["x"],
            "y": agent_position["y"],
            "z": agent_position["z"]
        }
        logger.info(f"Reference frame set to: ({self.reference_pos['x']:.2f}, {self.reference_pos['y']:.2f})")

    def reset_map(self):
        """Reset the entire map, reference frame, and create new episode directory."""
        self.map_data.fill(0.0)
        self.reference_pos = None  # Will be reset on next update
        
        # Setup new episode directory
        self._setup_episode_directory()
        
        logger.info(f"MindMap reset - Episode {self.episode_counter} started, reference frame will be set on next update")
        if self.enable_visualization:
            self._trigger_update()

    def update_from_raycast(self, agent_position: Dict[str, float], raycast_info: Dict[str, Any]):
        """Update the map based on agent position and raycast data."""
        try:
            # Update agent position
            self.agent_pos = agent_position.copy()
            
            # Set reference frame if not set (first frame or after reset)
            if self.reference_pos is None:
                self.set_reference_frame(agent_position)
            
            # Get agent position in map coordinates (relative to reference)
            agent_map_x, agent_map_y = self.world_to_map(agent_position["x"], agent_position["y"])
            
            # Check if agent is within map bounds
            if not self.is_valid_coordinate(agent_map_x, agent_map_y):
                logger.warning(f"Agent position out of map bounds: {agent_position} -> map coords: ({agent_map_x}, {agent_map_y})")
                
                # Still update frame counter and visualization
                self.frame_counter += 1
                if self.enable_visualization:
                    self._trigger_update()
                return
            
            # Apply decay to all channels except explored
            self.map_data[:6] *= self.decay_factor
            
            # Mark agent's current position as explored
            self.map_data[6, agent_map_y, agent_map_x] = 1.0
            
            # Process raycast data
            distances = raycast_info.get("distances", [])
            angles = raycast_info.get("angles", [])
            types = raycast_info.get("types", [])
            
            for distance, angle, obj_type in zip(distances, angles, types):
                if obj_type == 0:  # Skip nil types
                    continue
                    
                # Convert angle to radians and calculate hit point
                angle_rad = np.deg2rad(angle)
                hit_x = agent_position["x"] + distance * np.cos(angle_rad)
                hit_y = agent_position["y"] + distance * np.sin(angle_rad)
                
                # Convert to map coordinates (relative to reference)
                hit_map_x, hit_map_y = self.world_to_map(hit_x, hit_y)
                
                if not self.is_valid_coordinate(hit_map_x, hit_map_y):
                    continue
                
                # Update the appropriate channel
                channel = self.type_to_channel.get(obj_type, -1)
                if channel >= 0:
                    self.map_data[channel, hit_map_y, hit_map_x] = 1.0
                
                # Mark the ray path as explored
                self._mark_ray_path_explored(agent_map_x, agent_map_y, hit_map_x, hit_map_y)
            
            # Update visualization every frame
            self.frame_counter += 1
            if self.enable_visualization:
                self._trigger_update()
                
        except Exception as e:
            logger.error(f"Error updating map from raycast: {e}")
    
    def _trigger_update(self):
        """Trigger a visualization update in a thread-safe way."""
        with self.viz_lock:
            self.update_flag = True
    
    def _mark_ray_path_explored(self, x0: int, y0: int, x1: int, y1: int):
        """Mark the path of a ray as explored using Bresenham's line algorithm."""
        points = self._bresenham_line(x0, y0, x1, y1)
        for x, y in points:
            if self.is_valid_coordinate(x, y):
                self.map_data[6, y, x] = max(self.map_data[6, y, x], 0.5)  # Explored but not occupied
    
    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm for marking explored paths."""
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
    
    def get_local_map(self, radius: int = 64) -> np.ndarray:
        """Get a local map centered on the agent's position."""
        agent_map_x, agent_map_y = self.world_to_map(self.agent_pos["x"], self.agent_pos["y"])
        
        # Calculate bounds
        x_start = max(0, agent_map_x - radius)
        x_end = min(self.map_size, agent_map_x + radius)
        y_start = max(0, agent_map_y - radius)
        y_end = min(self.map_size, agent_map_y + radius)
        
        # Extract local map
        local_map = self.map_data[:, y_start:y_end, x_start:x_end]
        
        # Pad if necessary to maintain consistent size
        target_size = 2 * radius
        if local_map.shape[1] != target_size or local_map.shape[2] != target_size:
            padded_map = np.zeros((self.num_channels, target_size, target_size), dtype=np.float32)
            
            # Calculate padding offsets
            y_offset = (target_size - local_map.shape[1]) // 2
            x_offset = (target_size - local_map.shape[2]) // 2
            
            # Fix: Ensure we don't exceed bounds when copying
            y_end_copy = min(y_offset + local_map.shape[1], target_size)
            x_end_copy = min(x_offset + local_map.shape[2], target_size)
            
            # Only copy the part that fits
            copy_height = y_end_copy - y_offset
            copy_width = x_end_copy - x_offset
            
            padded_map[:, 
                      y_offset:y_end_copy, 
                      x_offset:x_end_copy] = local_map[:, :copy_height, :copy_width]
            return padded_map
        
        return local_map
    
    def get_map_tensor(self, device: torch.device = None) -> torch.Tensor:
        """Get the full map as a PyTorch tensor."""
        tensor = torch.from_numpy(self.map_data.copy())
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    
    def get_map_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the map."""
        channel_names = ["obstacle", "enemy", "trap", "goal", "people", "food", "explored"]
        summary = {}
        
        for i, name in enumerate(channel_names):
            channel_data = self.map_data[i]
            summary[name] = {
                "max_value": float(np.max(channel_data)),
                "total_coverage": float(np.sum(channel_data > 0.1)),
                "density": float(np.mean(channel_data))
            }
        
        summary["agent_position"] = self.agent_pos.copy()
        return summary
    
    def close_visualization(self):
        """Close the matplotlib visualization."""
        if self.enable_visualization and self.viz_running:
            self.viz_running = False
            if self.viz_thread and self.viz_thread.is_alive():
                self.viz_thread.join(timeout=2.0)
            if self.fig:
                plt.close(self.fig)
            logger.info("Visualization closed")
    
    def _save_frame(self):
        """Save current frame as PNG in current episode directory."""
        try:
            if not self.current_episode_dir:
                logger.warning("No episode directory set - cannot save frame")
                return
                
            filename = f"mindmap_frame_{self.frame_counter:06d}.png"
            filepath = self.current_episode_dir / filename
            
            # Save with high quality
            self.fig.savefig(
                filepath,
                dpi=100,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            
            # Log every frame for debugging (change back to % 50 later)
            logger.info(f"Episode {self.episode_counter}: Saved frame: {filename}")
                
        except Exception as e:
            logger.error(f"Error saving frame {self.frame_counter} in episode {self.episode_counter}: {e}")
    
    def get_frame_count(self) -> int:
        """Get total number of saved frames in current episode."""
        if not self.save_frames or not self.current_episode_dir or not self.current_episode_dir.exists():
            return 0
        return len(list(self.current_episode_dir.glob("mindmap_frame_*.png")))
    
    def create_gif_from_frames(self, gif_path: str = None, fps: int = 10, max_frames: int = None):
        """Create a GIF animation from saved PNG frames in current episode."""
        try:
            from PIL import Image
            
            if not self.current_episode_dir or not self.current_episode_dir.exists():
                logger.warning("No current episode directory found")
                return
            
            # Default GIF name includes episode number
            if gif_path is None:
                gif_path = f"episode_{self.episode_counter:04d}_animation.gif"
            
            # Get all frame files from current episode
            frame_files = sorted(self.current_episode_dir.glob("mindmap_frame_*.png"))
            
            if max_frames:
                frame_files = frame_files[-max_frames:]  # Take last N frames
            
            if not frame_files:
                logger.warning(f"No frames found in episode {self.episode_counter} to create GIF")
                return
            
            # Load images
            images = []
            for frame_file in frame_files:
                img = Image.open(frame_file)
                images.append(img)
            
            # Save as GIF in episode directory
            gif_output = self.current_episode_dir / gif_path
            images[0].save(
                gif_output,
                save_all=True,
                append_images=images[1:],
                duration=1000//fps,  # milliseconds per frame
                loop=0
            )
            
            logger.info(f"Created GIF: {gif_output} with {len(images)} frames")
            
        except ImportError:
            logger.error("PIL/Pillow required for GIF creation: pip install Pillow")
        except Exception as e:
            logger.error(f"Error creating GIF for episode {self.episode_counter}: {e}")
    
    def cleanup_old_episodes(self, keep_last_n: int = 10):
        """Keep only the last N episodes to prevent disk space issues."""
        if not self.save_frames or not self.base_output_dir.exists():
            return
        
        episode_dirs = [d for d in self.base_output_dir.iterdir() 
                       if d.is_dir() and d.name.startswith("episode_")]
        
        if len(episode_dirs) > keep_last_n:
            # Sort by episode number
            episode_dirs.sort(key=lambda x: int(x.name.split("_")[1]))
            dirs_to_delete = episode_dirs[:-keep_last_n]
            
            for episode_dir in dirs_to_delete:
                try:
                    # Delete all files in the episode directory
                    for file_path in episode_dir.iterdir():
                        file_path.unlink()
                    # Delete the directory
                    episode_dir.rmdir()
                except Exception as e:
                    logger.error(f"Error deleting episode directory {episode_dir}: {e}")
            
            logger.info(f"Cleaned up {len(dirs_to_delete)} old episodes, kept last {keep_last_n}")

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of all episodes."""
        if not self.base_output_dir.exists():
            return {"total_episodes": 0, "episodes": []}
        
        episode_dirs = [d for d in self.base_output_dir.iterdir() 
                       if d.is_dir() and d.name.startswith("episode_")]
        
        episodes = []
        for ep_dir in episode_dirs:
            try:
                ep_num = int(ep_dir.name.split("_")[1])
                frame_count = len(list(ep_dir.glob("mindmap_frame_*.png")))
                episodes.append({
                    "episode": ep_num,
                    "directory": str(ep_dir),
                    "frame_count": frame_count
                })
            except (IndexError, ValueError):
                continue
        
        episodes.sort(key=lambda x: x["episode"])
        
        return {
            "total_episodes": len(episodes),
            "current_episode": self.episode_counter,
            "episodes": episodes
        }

    def is_valid_coordinate(self, map_x: int, map_y: int) -> bool:
        """Check if map coordinates are within bounds."""
        return 0 <= map_x < self.map_size and 0 <= map_y < self.map_size


class ContextParser:
    
    def __init__(self, device=None, map_size: int = 512, map_resolution: float = 0.5, enable_viz: bool = True, save_frames: bool = False, output_dir: str = "mindmap_frames"):
        self.device = device or torch.device('cpu')
        self.mind_map = MindMap(
            map_size=map_size, 
            resolution=map_resolution, 
            enable_visualization=enable_viz,
            save_frames=save_frames,
            output_dir=output_dir
        )
    
    def parse_context(self, raw_data: Dict[str, Any], perceive_vision_fn=None, construct_vision_fn=None) -> Dict[str, Any]:
        try:
            line_of_sight = raw_data.get("line_of_sight", [])

            agent_position = {
                "x": raw_data.get("x", 0.0),
                "y": raw_data.get("y", 0.0),
                "z": raw_data.get("z", 0.0),
            }
            agent_state = raw_data.get("state", 0)
            hit_point = raw_data.get("hit_point", 100)
            hunger = raw_data.get("hunger", 0.0)

            # Reset mind map AND reference frame if agent state equals 3
            if agent_state == 3:
                self.mind_map.reset_map()
                logger.info("Agent state = 3: MindMap and reference frame reset")

            raycast_info = process_line_of_sight(line_of_sight)
            raycast_matrices = create_raycast_matrices(raycast_info)

            # Update mind map with current raycast data
            self.mind_map.update_from_raycast(agent_position, raycast_info)

            # Get mind map data instead of vision latent
            mind_map_tensor = self.mind_map.get_map_tensor(self.device)
            local_mind_map = self.mind_map.get_local_map(radius=64)
            local_mind_map_tensor = torch.from_numpy(local_mind_map).to(self.device)

            lidar_vision_tensor = create_lidar_vision_tensor(raycast_matrices)
            detection_summary = self.categorize_detections(raycast_info)

            parsed_context = {
                "raycast_info": raycast_info,
                "raycast_matrices": raycast_matrices,
                "mind_map": mind_map_tensor,
                "local_mind_map": local_mind_map_tensor,
                "mind_map_summary": self.mind_map.get_map_summary(),
                "lidar_vision_tensor": lidar_vision_tensor,
                "agent_position": agent_position,
                "agent_state": agent_state,
                "hit_point": hit_point,
                "hunger": hunger,
                "detection_summary": detection_summary,
                "raw_perception": raw_data,
            }

            return parsed_context

        except Exception as e:
            logger.error(f"Failed to parse context from raw data: {e}")
            logger.exception("Context parsing error")
            logger.debug(f"Raw data received: {raw_data}")
            
            # Return a fallback context with required keys to prevent downstream errors
            fallback_context = {
                "raycast_info": {"distances": [], "angles": [], "types": []},
                "raycast_matrices": {"obstacle_matrix": torch.zeros((0, 2)), "enemy_matrix": torch.zeros((0, 2)), "empty_matrix": torch.zeros((0, 2))},
                "mind_map": torch.zeros((self.mind_map.num_channels, self.mind_map.map_size, self.mind_map.map_size)),
                "local_mind_map": torch.zeros((self.mind_map.num_channels, 128, 128)),
                "mind_map_summary": {"agent_position": {"x": 0.0, "y": 0.0, "z": 0.0}},
                "lidar_vision_tensor": torch.zeros((3, 64, 64)),
                "agent_position": {"x": raw_data.get("x", 0.0), "y": raw_data.get("y", 0.0), "z": raw_data.get("z", 0.0)},
                "agent_state": raw_data.get("state", 0),
                "hit_point": raw_data.get("hit_point", 100),
                "hunger": raw_data.get("hunger", 0.0),
                "detection_summary": {"nil": {"count": 0, "distances": [], "angles": []}},
                "raw_perception": raw_data,
            }
            
            return fallback_context

    def categorize_detections(self, raycast_info: Dict[str, Any]) -> Dict[str, Any]:
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
            if type_name != "nil":
                categorized[type_name]["count"] += 1
                categorized[type_name]["distances"].append(distance)
                categorized[type_name]["angles"].append(angle)

        return categorized
    
    def cleanup(self):
        """Clean up resources including visualization."""
        self.mind_map.close_visualization()


class VisionConstructor:
    
    @staticmethod
    def construct_vision_from_raycast(line_of_sight: List[Dict[str, Any]]) -> np.ndarray:
        max_distance = 100.0
        height = 64
        width = 64

        vision_matrix = np.zeros((height, width, 3), dtype=np.float32)

        if not line_of_sight:
            return vision_matrix

        num_rays = len(line_of_sight)

        for i, ray in enumerate(line_of_sight):
            distance = ray.get("Distance", max_distance)
            obj_type = ray.get("Type", 0)

            depth_value = 1.0 - min(distance / max_distance, 1.0)
            col = int((i / max(num_rays - 1, 1)) * (width - 1))
            row = int((1.0 - depth_value) * (height - 1))

            vision_matrix[row, col, 0] = depth_value

            type_intensity = min(obj_type / 6.0, 1.0) if obj_type > 0 else 0.0
            vision_matrix[row, col, 1] = type_intensity

            vision_matrix[row, col, 2] = (depth_value + type_intensity) / 2.0

            for r_offset in range(-3, 4):
                new_row = row + r_offset
                if 0 <= new_row < height:
                    fade_factor = max(0, 1.0 - abs(r_offset) * 0.2)

                    for channel in range(3):
                        current_val = vision_matrix[new_row, col, channel]
                        new_val = vision_matrix[row, col, channel] * fade_factor
                        vision_matrix[new_row, col, channel] = max(current_val, new_val)

        kernel_size = 3
        for channel in range(3):
            smoothed_channel = np.copy(vision_matrix[:, :, channel])
            for row in range(kernel_size // 2, height - kernel_size // 2):
                for col in range(kernel_size // 2, width - kernel_size // 2):
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

        vision_matrix = (vision_matrix * 255.0).astype(np.uint8)

        return vision_matrix

    @staticmethod
    def construct_polar_vision_from_raycast(line_of_sight: List[Dict[str, Any]]) -> np.ndarray:
        height = 64
        width = 64
        max_distance = 100.0

        vision_matrix = np.zeros((height, width, 3), dtype=np.float32)

        if not line_of_sight:
            return (vision_matrix * 255.0).astype(np.uint8)

        num_rays = len(line_of_sight)
        center_x, center_y = width // 2, height // 2

        for i, ray in enumerate(line_of_sight):
            distance = ray.get("Distance", max_distance)
            obj_type = ray.get("Type", 0)

            angle = (i / max(num_rays - 1, 1)) * 2 * np.pi
            normalized_distance = min(distance / max_distance, 1.0)

            radius = normalized_distance * min(center_x, center_y)
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))

            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))

            intensity = 1.0 - normalized_distance
            type_intensity = min(obj_type / 6.0, 1.0) if obj_type > 0 else 0.0

            vision_matrix[y, x, 0] = intensity
            vision_matrix[y, x, 1] = type_intensity
            vision_matrix[y, x, 2] = (intensity + type_intensity) / 2.0

            line_points = VisionConstructor._bresenham_line(center_x, center_y, x, y)
            for lx, ly in line_points:
                if 0 <= lx < width and 0 <= ly < height:
                    fade = max(0, intensity * 0.3)
                    vision_matrix[ly, lx, 0] = max(vision_matrix[ly, lx, 0], fade)

        return (vision_matrix * 255.0).astype(np.uint8)

    @staticmethod
    def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[tuple]:
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