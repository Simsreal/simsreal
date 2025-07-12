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
from loguru import logger


class MindMap:
    """
    2D occupancy map that tracks environment features based on agent's raycast data.
    Updates continuously and resets when agent state equals 3.
    """

    def __init__(
        self,
        map_size: int = 512,
        resolution: float = 0.5,
        decay_factor: float = 0.99,
        enable_visualization: bool = True,
        save_frames: bool = False,
        output_dir: str = "mindmap_frames",
        environment_type: str = "auto",
    ):
        """Initialize the MindMap with reference frame system."""
        self.map_size = map_size
        self.resolution = resolution
        self.decay_factor = decay_factor
        self.enable_visualization = enable_visualization
        self.save_frames = save_frames
        self.base_output_dir = Path(output_dir)
        self.environment_type = environment_type

        # Episode management
        self.episode_counter = 0
        self.current_episode_dir = None
        self._setup_episode_directory()

        # Map channels: [obstacle, checkpoint, trap, goal, people, food, explored]
        self.num_channels = 7
        self.map_data = np.zeros(
            (self.num_channels, map_size, map_size), dtype=np.float32
        )

        # Type mapping from raycast to channels
        self.type_to_channel = {
            0: -1,  # nil - no update
            1: 0,  # obstacle
            2: 1,  # checkpoint (was enemy)
            3: 2,  # trap
            4: 3,  # goal
            5: 4,  # people
            6: 5,  # food
        }

        # Improved colors for better visualization
        self.colors = {
            0: [0.4, 0.4, 0.4],  # obstacle - dark gray
            1: [0.0, 0.7, 1.0],  # checkpoint - bright blue
            2: [1.0, 0.3, 0.0],  # trap - orange-red
            3: [0.0, 0.8, 0.2],  # goal - bright green
            4: [1.0, 0.0, 1.0],  # people - magenta
            5: [0.0, 1.0, 1.0],  # food - cyan
            6: [0.15, 0.15, 0.15],  # explored - very dark gray
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
        self.direction_arrow = None
        self.viz_thread = None
        self.viz_running = False
        self.update_flag = False
        self.viz_lock = threading.Lock()

        if self.enable_visualization:
            self._init_matplotlib_visualization()

        logger.info(
            f"MindMap initialized: {map_size}x{map_size}, resolution={resolution}m/px, "
            f"environment_type={environment_type}, reference frame system enabled"
        )

    def _setup_episode_directory(self):
        """Setup directory for current episode."""
        if self.save_frames:
            # Create base directory if it doesn't exist
            self.base_output_dir.mkdir(exist_ok=True)

            # Find the next episode number
            existing_episodes = [
                d
                for d in self.base_output_dir.iterdir()
                if d.is_dir() and d.name.startswith("episode_")
            ]

            if existing_episodes:
                episode_numbers = []
                for ep_dir in existing_episodes:
                    try:
                        ep_num = int(ep_dir.name.split("_")[1])
                        episode_numbers.append(ep_num)
                    except (IndexError, ValueError):
                        continue
                self.episode_counter = (
                    max(episode_numbers) + 1 if episode_numbers else 0
                )
            else:
                self.episode_counter = 0

            # Create current episode directory
            self.current_episode_dir = (
                self.base_output_dir / f"episode_{self.episode_counter:04d}"
            )
            self.current_episode_dir.mkdir(exist_ok=True)

            # Reset frame counter for new episode
            self.frame_counter = 0

            # Removed episode setup logging to reduce log noise

    def _init_matplotlib_visualization(self):
        """Initialize matplotlib visualization in a separate thread."""
        try:
            self.viz_running = True
            self.viz_thread = threading.Thread(
                target=self._visualization_loop, daemon=True
            )
            self.viz_thread.start()
            logger.info("Matplotlib visualization thread started")
        except Exception as e:
            logger.error(f"Failed to start visualization thread: {e}")
            self.enable_visualization = False

    def _visualization_loop(self):
        """Main visualization loop with improved UI."""
        try:
            # Use Agg backend for PNG generation (no GUI required)
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Create figure with better layout
            self.fig, self.ax = plt.subplots(figsize=(14, 10), dpi=120)
            self.fig.patch.set_facecolor('black')
            
            # Style the main plot
            self.ax.set_facecolor('black')
            self.ax.set_title("SimsReal Agent MindMap", fontsize=16, color='white', fontweight='bold')
            self.ax.set_xlabel("Map X (pixels)", fontsize=12, color='white')
            self.ax.set_ylabel("Map Z (pixels)", fontsize=12, color='white')
            
            # Style tick labels
            self.ax.tick_params(colors='white', labelsize=10)
            for spine in self.ax.spines.values():
                spine.set_color('white')

            # Create initial empty image
            initial_img = np.zeros((self.map_size, self.map_size, 3))
            self.im = self.ax.imshow(
                initial_img, origin="lower", extent=[0, self.map_size, 0, self.map_size]
            )

            # Add improved legend
            self._add_improved_legend()

            # Initial agent marker (larger and more visible)
            (self.agent_marker,) = self.ax.plot(
                [], [], "o", color='white', markersize=12, 
                markeredgecolor="yellow", markeredgewidth=3,
                markerfacecolor='white', alpha=0.9
            )
            
            # Direction arrow for agent orientation
            self.direction_arrow = self.ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                                  arrowprops=dict(arrowstyle='->', 
                                                                color='yellow', 
                                                                lw=3, alpha=0.8))

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

    def _add_improved_legend(self):
        """Add an improved legend with better styling."""
        legend_elements = []
        legend_labels = [
            "Obstacle",
            "Checkpoint", 
            "Trap",
            "Goal",
            "People",
            "Food",
            "Explored",
            "Agent",
        ]
        legend_colors = [self.colors[i] for i in range(7)] + [
            [1.0, 1.0, 1.0]
        ]  # White for agent

        for label, color in zip(legend_labels, legend_colors):
            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, label=label, 
                            edgecolor='white', linewidth=0.5)
            )

        legend = self.ax.legend(
            handles=legend_elements, 
            loc="upper left", 
            bbox_to_anchor=(1.02, 1),
            fontsize=11,
            facecolor='black',
            edgecolor='white',
            labelcolor='white'
        )
        legend.get_frame().set_alpha(0.8)

    def _update_plot(self):
        """Update the matplotlib plot with current map data and improved visuals."""
        try:
            # Create RGB image from map data with better blending
            viz_img = np.zeros((self.map_size, self.map_size, 3))

            # Render each channel with its color (priority order: objects > explored)
            for channel_idx in [6, 0, 1, 2, 3, 4, 5]:  # explored first, then objects
                channel_data = self.map_data[channel_idx]
                mask = channel_data > 0.1

                if np.any(mask):
                    color = self.colors[channel_idx]
                    # Use alpha blending for better visualization
                    alpha = channel_data[mask] if channel_idx == 6 else np.ones(np.sum(mask))
                    for i in range(3):
                        viz_img[mask, i] = (1 - alpha) * viz_img[mask, i] + alpha * color[i]

            # Update image
            self.im.set_data(viz_img)

            # Update agent position using X,Z coordinates
            agent_map_x, agent_map_y = self.world_to_map(
                self.agent_pos["x"], self.agent_pos["z"]  # Use Z coordinate
            )
            if self.is_valid_coordinate(agent_map_x, agent_map_y):
                self.agent_marker.set_data([agent_map_x], [agent_map_y])
                
                # Update direction arrow
                orientation = self.agent_pos.get('orientation', 0.0)
                arrow_length = 15  # pixels
                arrow_end_x = agent_map_x + arrow_length * np.sin(np.deg2rad(orientation))
                arrow_end_y = agent_map_y + arrow_length * np.cos(np.deg2rad(orientation))
                
                self.direction_arrow.set_position((agent_map_x, agent_map_y))
                self.direction_arrow.xy = (arrow_end_x, arrow_end_y)

            # Update title with improved formatting and more info
            timestamp = time.strftime("%H:%M:%S")
            if self.reference_pos:
                rel_x = self.agent_pos["x"] - self.reference_pos["x"]
                rel_z = self.agent_pos["z"] - self.reference_pos["z"]  # Use Z coordinate
                
                # Calculate map coverage
                explored_coverage = np.sum(self.map_data[6] > 0.1) / (self.map_size * self.map_size) * 100
                
                title = (f"Episode {self.episode_counter} | Frame: {self.frame_counter} | "
                        f"Agent: World({self.agent_pos['x']:.1f}, {self.agent_pos['z']:.1f}) "
                        f"Rel({rel_x:.1f}, {rel_z:.1f}) | "
                        f"Heading: {self.agent_pos.get('orientation', 0):.0f}° | "
                        f"Explored: {explored_coverage:.1f}% | {timestamp}")
            else:
                title = (f"Episode {self.episode_counter} | Frame: {self.frame_counter} | "
                        f"Agent: ({self.agent_pos['x']:.1f}, {self.agent_pos['z']:.1f}) | "
                        f"Heading: {self.agent_pos.get('orientation', 0):.0f}° | {timestamp}")

            self.ax.set_title(title, fontsize=12, color='white', fontweight='bold')

        except Exception as e:
            logger.error(f"Error updating plot: {e}")

    def world_to_map(self, world_x: float, world_z: float) -> Tuple[int, int]:
        """Convert world coordinates to map indices using reference frame.
        Note: Unity uses X,Z for 2D plane, so world_z is treated as map Y coordinate.
        """
        if self.reference_pos is None:
            # If no reference set, use current position as reference (centered)
            map_x = self.map_center
            map_y = self.map_center
        else:
            # Calculate relative position from reference
            rel_x = world_x - self.reference_pos["x"]
            rel_z = world_z - self.reference_pos["z"]  # Use Z coordinate

            # Convert to map coordinates (reference pos is at map center)
            # X maps to map X, Z maps to map Y
            map_x = int(self.map_center + rel_x / self.resolution)
            map_y = int(self.map_center + rel_z / self.resolution)

        return map_x, map_y

    def map_to_world(self, map_x: int, map_y: int) -> Tuple[float, float]:
        """Convert map indices to world coordinates using reference frame.
        Returns (world_x, world_z) since Unity uses X,Z for 2D plane.
        """
        if self.reference_pos is None:
            return 0.0, 0.0

        # Calculate relative position from map center
        rel_x = (map_x - self.map_center) * self.resolution
        rel_z = (map_y - self.map_center) * self.resolution  # Map Y corresponds to world Z

        # Add to reference position to get world coordinates
        world_x = self.reference_pos["x"] + rel_x
        world_z = self.reference_pos["z"] + rel_z  # Return Z coordinate

        return world_x, world_z

    def set_reference_frame(self, agent_position: Dict[str, float]):
        """Set the reference frame based on agent position."""
        if self.environment_type == "racing" or (self.environment_type == "auto" and self._detect_racing_environment()):
            # Place reference frame so agent starts in lower third of map
            # This means the reference point should be offset forward (positive Z) from agent position
            offset_z = self.map_size * self.resolution * 0.3  # 30% of map size forward
            self.reference_pos = {
                "x": agent_position["x"],
                "y": agent_position["y"],  # Keep Y as-is for height
                "z": agent_position["z"] + offset_z,  # Offset reference forward in Z
            }
            # Removed reference frame setup logging to reduce log noise
        else:
            # Default behavior - agent at center
            self.reference_pos = {
                "x": agent_position["x"],
                "y": agent_position["y"],
                "z": agent_position["z"],
            }
            # Removed reference frame setup logging to reduce log noise

    def _detect_racing_environment(self) -> bool:
        """Auto-detect if this is a racing environment based on raycast data patterns."""
        # Simple heuristic: if we consistently see goals/traps in a forward direction
        # This is a placeholder - could be enhanced with more sophisticated detection
        return True  # For now, assume racing environment

    def reset_map(self):
        """Reset the entire map, reference frame, and create new episode directory."""
        # Clean up previous episode if it has too few frames
        self._cleanup_current_episode_if_insufficient()
        
        self.map_data.fill(0.0)
        self.reference_pos = None  # Will be reset on next update

        # Setup new episode directory
        self._setup_episode_directory()

        # Removed map reset logging to reduce log noise
        if self.enable_visualization:
            self._trigger_update()

    def _cleanup_current_episode_if_insufficient(self, min_frames: int = 10):
        """Remove current episode directory if it has fewer than min_frames."""
        if not self.save_frames or not self.current_episode_dir:
            return
            
        if not self.current_episode_dir.exists():
            return
            
        # Count frames in current episode
        frame_count = len(list(self.current_episode_dir.glob("mindmap_frame_*.png")))
        
        if frame_count < min_frames:
            try:
                # Delete all files in the episode directory
                for file_path in self.current_episode_dir.iterdir():
                    file_path.unlink()
                # Delete the directory
                self.current_episode_dir.rmdir()
                # Removed episode deletion logging to reduce log noise
            except Exception as e:
                logger.error(f"Error deleting insufficient episode directory {self.current_episode_dir}: {e}")
        else:
            # Removed episode deletion logging to reduce log noise
            pass

    def update_from_raycast(
        self, agent_position: Dict[str, float], raycast_info: Dict[str, Any]
    ):
        """Update the map based on agent position and raycast data."""
        try:
            # Update agent position
            self.agent_pos = agent_position.copy()

            # Set reference frame if not set (first frame or after reset)
            if self.reference_pos is None:
                self.set_reference_frame(agent_position)

            # Get agent position in map coordinates using X,Z coordinates
            agent_map_x, agent_map_y = self.world_to_map(
                agent_position["x"], agent_position["z"]  # Use Z instead of Y
            )

            # Check if agent is within map bounds
            if not self.is_valid_coordinate(agent_map_x, agent_map_y):
                logger.warning(
                    f"Agent position out of map bounds: {agent_position} -> map coords: ({agent_map_x}, {agent_map_y})"
                )

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
            
            # Get agent orientation from position data
            agent_orientation = agent_position.get("orientation", 0.0)

            # Collect raycast endpoints for FOV filling
            raycast_endpoints = []

            for distance, angle, obj_type in zip(distances, angles, types):
                # Calculate world angle: agent orientation + relative raycast angle
                # The raycast angles are relative to agent's facing direction
                world_angle = agent_orientation + angle
                angle_rad = np.deg2rad(world_angle)
                
                # Calculate hit position in world coordinates
                # In Unity: X is left-right, Z is forward-back (2D plane)
                hit_x = agent_position["x"] + distance * np.sin(angle_rad)  # X component
                hit_z = agent_position["z"] + distance * np.cos(angle_rad)  # Z component

                # Convert to map coordinates (X,Z -> map_x,map_y)
                hit_map_x, hit_map_y = self.world_to_map(hit_x, hit_z)

                if not self.is_valid_coordinate(hit_map_x, hit_map_y):
                    continue

                # Store endpoint for FOV filling
                raycast_endpoints.append((hit_map_x, hit_map_y, angle))

                # Update the appropriate channel for obstacles/objects (not nil)
                if obj_type != 0:
                    channel = self.type_to_channel.get(obj_type, -1)
                    if channel >= 0:
                        self.map_data[channel, hit_map_y, hit_map_x] = 1.0

                # Mark the ray path as explored
                self._mark_ray_path_explored(
                    agent_map_x, agent_map_y, hit_map_x, hit_map_y
                )

            # Fill the fan-shaped FOV area
            self._fill_fov_fan(agent_map_x, agent_map_y, raycast_endpoints)

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
                self.map_data[6, y, x] = max(
                    self.map_data[6, y, x], 0.5
                )  # Explored but not occupied

    def _bresenham_line(
        self, x0: int, y0: int, x1: int, y1: int
    ) -> List[Tuple[int, int]]:
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
        agent_map_x, agent_map_y = self.world_to_map(
            self.agent_pos["x"], self.agent_pos["z"]  # Use Z coordinate
        )

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
            padded_map = np.zeros(
                (self.num_channels, target_size, target_size), dtype=np.float32
            )

            # Calculate padding offsets
            y_offset = (target_size - local_map.shape[1]) // 2
            x_offset = (target_size - local_map.shape[2]) // 2

            # Fix: Ensure we don't exceed bounds when copying
            y_end_copy = min(y_offset + local_map.shape[1], target_size)
            x_end_copy = min(x_offset + local_map.shape[2], target_size)

            # Only copy the part that fits
            copy_height = y_end_copy - y_offset
            copy_width = x_end_copy - x_offset

            padded_map[:, y_offset:y_end_copy, x_offset:x_end_copy] = local_map[
                :, :copy_height, :copy_width
            ]
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
        channel_names = [
            "obstacle",
            "checkpoint",  # Changed from "enemy"
            "trap",
            "goal",
            "people",
            "food",
            "explored",
        ]
        summary = {}

        for i, name in enumerate(channel_names):
            channel_data = self.map_data[i]
            summary[name] = {
                "max_value": float(np.max(channel_data)),
                "total_coverage": float(np.sum(channel_data > 0.1)),
                "density": float(np.mean(channel_data)),
            }

        summary["agent_position"] = self.agent_pos.copy()
        return summary

    def close_visualization(self):
        """Close the matplotlib visualization."""
        # Clean up current episode before closing
        self._cleanup_current_episode_if_insufficient()
        
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
                dpi=120,
                bbox_inches="tight",
                facecolor="black",
                edgecolor="none",
            )

            # Removed excessive per-frame logging to reduce log noise

        except Exception as e:
            logger.error(
                f"Error saving frame {self.frame_counter} in episode {self.episode_counter}: {e}"
            )

    def get_frame_count(self) -> int:
        """Get total number of saved frames in current episode."""
        if (
            not self.save_frames
            or not self.current_episode_dir
            or not self.current_episode_dir.exists()
        ):
            return 0
        return len(list(self.current_episode_dir.glob("mindmap_frame_*.png")))

    def create_gif_from_frames(
        self, gif_path: str = None, fps: int = 10, max_frames: int = None
    ):
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
                logger.warning(
                    f"No frames found in episode {self.episode_counter} to create GIF"
                )
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
                duration=1000 // fps,  # milliseconds per frame
                loop=0,
            )

            logger.info(f"Created GIF: {gif_output} with {len(images)} frames")

        except ImportError:
            logger.error("PIL/Pillow required for GIF creation: pip install Pillow")
        except Exception as e:
            logger.error(f"Error creating GIF for episode {self.episode_counter}: {e}")

    def cleanup_old_episodes(self, keep_last_n: int = 10, min_frames: int = 10):
        """Keep only the last N episodes and remove episodes with insufficient frames."""
        if not self.save_frames or not self.base_output_dir.exists():
            return

        episode_dirs = [
            d
            for d in self.base_output_dir.iterdir()
            if d.is_dir() and d.name.startswith("episode_")
        ]

        # First pass: remove episodes with insufficient frames
        dirs_to_check = []
        for episode_dir in episode_dirs:
            frame_count = len(list(episode_dir.glob("mindmap_frame_*.png")))
            if frame_count < min_frames:
                try:
                    # Delete all files in the episode directory
                    for file_path in episode_dir.iterdir():
                        file_path.unlink()
                    # Delete the directory
                    episode_dir.rmdir()
                    # Removed episode deletion logging to reduce log noise
                except Exception as e:
                    logger.error(f"Error deleting insufficient episode directory {episode_dir}: {e}")
            else:
                dirs_to_check.append(episode_dir)

        # Second pass: keep only last N episodes from remaining valid episodes
        if len(dirs_to_check) > keep_last_n:
            # Sort by episode number
            dirs_to_check.sort(key=lambda x: int(x.name.split("_")[1]))
            dirs_to_delete = dirs_to_check[:-keep_last_n]

            for episode_dir in dirs_to_delete:
                try:
                    # Delete all files in the episode directory
                    for file_path in episode_dir.iterdir():
                        file_path.unlink()
                    # Delete the directory
                    episode_dir.rmdir()
                except Exception as e:
                    logger.error(f"Error deleting old episode directory {episode_dir}: {e}")

            # Removed episode deletion logging to reduce log noise

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of all episodes."""
        if not self.base_output_dir.exists():
            return {"total_episodes": 0, "episodes": []}

        episode_dirs = [
            d
            for d in self.base_output_dir.iterdir()
            if d.is_dir() and d.name.startswith("episode_")
        ]

        episodes = []
        for ep_dir in episode_dirs:
            try:
                ep_num = int(ep_dir.name.split("_")[1])
                frame_count = len(list(ep_dir.glob("mindmap_frame_*.png")))
                episodes.append(
                    {
                        "episode": ep_num,
                        "directory": str(ep_dir),
                        "frame_count": frame_count,
                    }
                )
            except (IndexError, ValueError):
                continue

        episodes.sort(key=lambda x: x["episode"])

        return {
            "total_episodes": len(episodes),
            "current_episode": self.episode_counter,
            "episodes": episodes,
        }

    def is_valid_coordinate(self, map_x: int, map_y: int) -> bool:
        """Check if map coordinates are within bounds."""
        return 0 <= map_x < self.map_size and 0 <= map_y < self.map_size

    def _fill_fov_fan(self, agent_x: int, agent_y: int, raycast_endpoints: List[Tuple[int, int, float]]):
        """Fill only the fan-shaped field of view area based on raycast coverage."""
        if len(raycast_endpoints) < 2:
            return
            
        # Sort endpoints by angle
        sorted_endpoints = sorted(raycast_endpoints, key=lambda x: x[2])  # Sort by angle
        
        # Fill triangular sectors between consecutive rays
        for i in range(len(sorted_endpoints)):
            curr_point = sorted_endpoints[i]
            next_point = sorted_endpoints[(i + 1) % len(sorted_endpoints)]
            
            # Only fill if the angular gap is reasonable (not wrapping around the full circle)
            angle_diff = abs(curr_point[2] - next_point[2])
            if angle_diff > 180:  # Handle wrap-around case
                angle_diff = 360 - angle_diff
                
            # Only fill sectors for adjacent rays (small angular gaps)
            if angle_diff < 45:  # Adjust this threshold as needed
                self._fill_fov_triangle(agent_x, agent_y, 
                                      curr_point[0], curr_point[1], 
                                      next_point[0], next_point[1])

    def _fill_fov_triangle(self, agent_x: int, agent_y: int, x1: int, y1: int, x2: int, y2: int):
        """Fill a triangular FOV sector using a simple scanline approach."""
        # Use a more efficient scanline fill for the triangle
        # This creates the fan-shaped visible area
        
        # Get all points in the triangle using simple geometric approach
        points = []
        
        # Add the three vertices
        points.append((agent_x, agent_y))
        points.append((x1, y1))
        points.append((x2, y2))
        
        # Find bounding box
        min_x = max(0, min(agent_x, x1, x2))
        max_x = min(self.map_size - 1, max(agent_x, x1, x2))
        min_y = max(0, min(agent_y, y1, y2))
        max_y = min(self.map_size - 1, max(agent_y, y1, y2))
        
        # Use barycentric coordinates to check if point is inside triangle
        def point_in_triangle(px, py, ax, ay, bx, by, cx, cy):
            denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(denom) < 1e-10:
                return False
            
            a = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
            b = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
            c = 1 - a - b
            
            return a >= 0 and b >= 0 and c >= 0
        
        # Fill all points inside the triangle
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if point_in_triangle(x, y, agent_x, agent_y, x1, y1, x2, y2):
                    # Mark as explored with medium confidence (FOV area)
                    self.map_data[6, y, x] = max(self.map_data[6, y, x], 0.6)

    # NEW METHOD: Save and load mindmap state
    def save_mindmap_state(self, filepath: str = None) -> str:
        """Save the current mindmap state to a file for later reuse."""
        try:
            if filepath is None:
                filepath = f"mindmap_state_episode_{self.episode_counter}.npz"
            
            # Save all relevant state
            np.savez_compressed(
                filepath,
                map_data=self.map_data,
                reference_pos=self.reference_pos,
                agent_pos=self.agent_pos,
                episode_counter=self.episode_counter,
                frame_counter=self.frame_counter,
                map_size=self.map_size,
                resolution=self.resolution,
                environment_type=self.environment_type
            )
            
            logger.info(f"Mindmap state saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving mindmap state: {e}")
            return None

    def load_mindmap_state(self, filepath: str) -> bool:
        """Load a previously saved mindmap state."""
        try:
            if not os.path.exists(filepath):
                logger.error(f"Mindmap state file not found: {filepath}")
                return False
                
            data = np.load(filepath, allow_pickle=True)
            
            # Restore state
            self.map_data = data['map_data']
            self.reference_pos = data['reference_pos'].item() if data['reference_pos'].ndim == 0 else data['reference_pos']
            self.agent_pos = data['agent_pos'].item() if data['agent_pos'].ndim == 0 else data['agent_pos']
            self.episode_counter = int(data['episode_counter'])
            self.frame_counter = int(data['frame_counter'])
            
            # Verify compatibility
            if (data['map_size'] != self.map_size or 
                data['resolution'] != self.resolution):
                logger.warning(f"Loaded mindmap has different parameters. "
                             f"Loaded: size={data['map_size']}, res={data['resolution']} "
                             f"Current: size={self.map_size}, res={self.resolution}")
            
            logger.info(f"Mindmap state loaded from: {filepath}")
            logger.info(f"Restored to episode {self.episode_counter}, frame {self.frame_counter}")
            
            # Trigger visualization update
            if self.enable_visualization:
                self._trigger_update()
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading mindmap state: {e}")
            return False


class MindmapBuilder:
    def __init__(
        self,
        device=None,
        map_size: int = 512,
        map_resolution: float = 0.5,
        enable_viz: bool = True,
        save_frames: bool = False,
        output_dir: str = "mindmap_frames",
        environment_type: str = "auto",  # "auto", "racing", "exploration"
    ):
        self.device = device or torch.device("cpu")
        self.environment_type = environment_type
        self.mind_map = MindMap(
            map_size=map_size,
            resolution=map_resolution,
            enable_visualization=enable_viz,
            save_frames=save_frames,
            output_dir=output_dir,
            environment_type=environment_type,
        )

    def parse_context(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            line_of_sight = raw_data.get("line_of_sight", [])

            agent_position = {
                "x": raw_data.get("x", 0.0),
                "y": raw_data.get("y", 0.0),  # Keep Y for height info
                "z": raw_data.get("z", 0.0),  # Z is the forward/back coordinate in Unity 2D plane
                "orientation": raw_data.get("orientation", 0.0),
            }
            agent_state = raw_data.get("state", 0)

            # Reset mind map AND reference frame if agent state equals 3
            if agent_state == 3:
                self.mind_map.reset_map()
                # Removed agent state reset logging to reduce log noise

            # Pass agent orientation to raycast processing
            raycast_info = process_line_of_sight(line_of_sight, agent_position["orientation"])

            # Update mind map with current raycast data
            self.mind_map.update_from_raycast(agent_position, raycast_info)

            # Get mind map data
            mind_map_tensor = self.mind_map.get_map_tensor(self.device)
            local_mind_map = self.mind_map.get_local_map(radius=64)
            local_mind_map_tensor = torch.from_numpy(local_mind_map).to(self.device)

            # Return context with raw context, raycast info, and mindmap data
            parsed_context = {
                "raw_context": raw_data,
                "raycast_info": raycast_info,
                "mind_map": mind_map_tensor,
                "local_mind_map": local_mind_map_tensor,
                "mind_map_summary": self.mind_map.get_map_summary(),
            }

            return parsed_context

        except Exception as e:
            logger.error(f"Failed to parse context from raw data: {e}")
            logger.exception("Context parsing error")
            logger.debug(f"Raw data received: {raw_data}")

            # Return a fallback context with required keys to prevent downstream errors
            fallback_context = {
                "raw_context": raw_data,
                "agent_state": 0,
                "raycast_info": {"distances": [], "angles": [], "types": []},
                "mind_map": torch.zeros(
                    (
                        self.mind_map.num_channels,
                        self.mind_map.map_size,
                        self.mind_map.map_size,
                    )
                ).to(self.device),
                "local_mind_map": torch.zeros((self.mind_map.num_channels, 128, 128)).to(self.device),
                "mind_map_summary": {"agent_position": {"x": 0.0, "y": 0.0, "z": 0.0}},
            }

            return fallback_context

    def save_state(self, filepath: str = None) -> str:
        """Save the current mindmap state."""
        return self.mind_map.save_mindmap_state(filepath)

    def load_state(self, filepath: str) -> bool:
        """Load a previously saved mindmap state."""
        return self.mind_map.load_mindmap_state(filepath)

    def cleanup(self):
        """Clean up resources including visualization."""
        # Clean up current episode before final cleanup
        self.mind_map._cleanup_current_episode_if_insufficient()
        self.mind_map.close_visualization()
