import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from manim import *
from loguru import logger
from simsreal_types import State, LineOfSight
import glob

# Import easing functions with fallbacks
try:
    from manim import ease_out_cubic, ease_in_cubic, smooth
except ImportError:
    # Fallback to basic rate functions if not available
    ease_out_cubic = lambda t: 1 - (1 - t) ** 3
    ease_in_cubic = lambda t: t ** 3
    smooth = lambda t: t


class SimsRealAnimation(Scene):
    def __init__(self, run_directory: str, **kwargs):
        super().__init__(**kwargs)
        self.run_directory = run_directory
        self.frames_data = []
        self.max_distance = 100.0
        self.scale_factor = 3.8  # Slightly smaller for better UI fit
        
    def load_frame_data(self):
        """Load all frame JSON data from the run directory"""
        json_files = sorted(glob.glob(os.path.join(self.run_directory, "frame_*.json")))
        
        # Sample frames to reduce animation complexity (every 3rd frame for smoother animation)
        sampled_files = json_files[::3]  # Take every 3rd frame
        
        for json_file in sampled_files:
            try:
                with open(json_file, 'r') as f:
                    frame_data = json.load(f)
                    self.frames_data.append(frame_data)
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {len(self.frames_data)} sampled frames from {self.run_directory}")
    
    def create_object_colors_and_sizes(self) -> Dict[int, Tuple[str, float, float]]:
        """Define colors, sizes, and stroke widths for different object types"""
        return {
            0: ("#9E9E9E", 0.06, 1.5),   # unknown - neutral gray
            1: ("#F44336", 0.03, 1),     # obstacle - red, very thin
            2: ("#4CAF50", 0.10, 2.5),   # checkpoint - green, medium
            3: ("#FF9800", 0.08, 2),     # trap - orange, medium-small
            4: ("#2196F3", 0.12, 3),     # goal - blue, large
            5: ("#FFEB3B", 0.09, 2.5),   # people - yellow, medium
            6: ("#E91E63", 0.07, 2)      # food - pink, small-medium
        }
    
    def state_to_objects(self, state: Dict) -> Tuple[List[Dot], List[Line]]:
        """Convert state data to Manim objects with improved styling"""
        dots = []
        lines = []
        object_styles = self.create_object_colors_and_sizes()
        
        # Agent position (center)
        agent_pos = np.array([0, 0, 0])
        
        line_of_sight = state.get('line_of_sight', [])
        num_rays = len(line_of_sight)
        
        if num_rays == 0:
            return dots, lines
        
        for i, ray in enumerate(line_of_sight):
            # Calculate angle (forward is up)
            angle = (i / num_rays) * 2 * np.pi - np.pi / 2
            
            # Normalize distance and apply scale
            normalized_distance = min(ray['distance'] / self.max_distance, 1.0)
            ray_length = normalized_distance * self.scale_factor
            
            # Calculate object position (Fix for Unity coordinate system)
            # Unity: Right = +X, Forward = +Z  
            # If right/left is inverted, flip the X coordinate
            obj_x = -ray_length * np.cos(angle)  # Flip X to match Unity
            obj_y = ray_length * np.sin(angle)   # Keep Y as is (forward = up)
            obj_pos = np.array([obj_x, obj_y, 0])
            
            # Get object style
            object_type = ray.get('type', 0)
            color, base_radius, stroke_width = object_styles.get(object_type, ("#9E9E9E", 0.06, 1.5))
            
            # Adjust size based on distance (closer = larger)
            distance_factor = 1.0 + (1.0 - normalized_distance) * 0.4
            dot_radius = base_radius * distance_factor
            
            # Create object dot with styling
            dot = Dot(
                point=obj_pos, 
                radius=dot_radius, 
                color=color,
                stroke_width=stroke_width,
                fill_opacity=0.85,
                stroke_opacity=0.9
            )
            
            # Add subtle glow for important objects
            if object_type in [2, 4, 6]:  # checkpoint, goal, food
                try:
                    glow = Circle(
                        radius=dot_radius * 1.5,
                        color=color,
                        fill_opacity=0.1,
                        stroke_opacity=0.3,
                        stroke_width=1
                    ).move_to(obj_pos)
                    dots.append(glow)
                except:
                    pass  # Skip glow if it causes issues
            
            dots.append(dot)
            
            # Create ray line with distance-based styling
            line_opacity = 0.2 + (1.0 - normalized_distance) * 0.3
            line_width = 0.5 + (1.0 - normalized_distance) * 0.3
            line = Line(
                agent_pos, obj_pos, 
                stroke_width=line_width, 
                color="#707070",
                stroke_opacity=line_opacity
            )
            lines.append(line)
        
        return dots, lines
    
    def create_enhanced_legend(self) -> VGroup:
        """Create a beautifully styled legend"""
        legend_group = VGroup()
        
        # Background panel - use Rectangle if RoundedRectangle not available
        try:
            bg_panel = RoundedRectangle(
                width=2.4, 
                height=4.2,
                corner_radius=0.1,
                color="#1A1A1A",
                fill_opacity=0.9,
                stroke_color="#333333",
                stroke_width=1.5
            )
        except:
            bg_panel = Rectangle(
                width=2.4, 
                height=4.2,
                color="#1A1A1A",
                fill_opacity=0.9,
                stroke_color="#333333",
                stroke_width=1.5
            )
        
        # Title
        title = Text("Objects", font_size=20, color="#FFFFFF")
        title.move_to(bg_panel.get_top() + DOWN * 0.3)
        
        # Object types
        object_styles = self.create_object_colors_and_sizes()
        type_names = {
            0: "Unknown", 1: "Obstacle", 2: "Checkpoint", 
            3: "Trap", 4: "Goal", 5: "People", 6: "Food"
        }
        
        legend_items = VGroup()
        y_pos = 1.4
        
        for obj_type, (color, radius, stroke_width) in object_styles.items():
            if obj_type in type_names:
                # Sample dot
                dot = Dot(
                    radius=radius * 1.3,
                    color=color,
                    stroke_width=stroke_width,
                    fill_opacity=0.85,
                    stroke_opacity=0.9
                )
                
                # Label
                label = Text(
                    type_names[obj_type], 
                    font_size=16, 
                    color="#E0E0E0"
                )
                
                # Arrange horizontally
                item_group = VGroup(dot, label)
                item_group.arrange(RIGHT, buff=0.25)
                item_group.move_to(UP * y_pos)
                
                legend_items.add(item_group)
                y_pos -= 0.5
        
        legend_content = VGroup(title, legend_items)
        legend_with_bg = VGroup(bg_panel, legend_content)
        legend_with_bg.shift(RIGHT * 4.5 + UP * 0.5)
        
        return legend_with_bg
    
    def create_info_panel(self, state: Dict, frame_num: int) -> VGroup:
        """Create an enhanced info panel"""
        info_group = VGroup()
        
        # Background panel
        try:
            bg_panel = RoundedRectangle(
                width=2.6, 
                height=3.5,
                corner_radius=0.1,
                color="#1A1A1A",
                fill_opacity=0.9,
                stroke_color="#333333",
                stroke_width=1.5
            )
        except:
            bg_panel = Rectangle(
                width=2.6, 
                height=3.5,
                color="#1A1A1A",
                fill_opacity=0.9,
                stroke_color="#333333",
                stroke_width=1.5
            )
        
        # Title
        title = Text("Agent Status", font_size=20, color="#FFFFFF")
        title.move_to(bg_panel.get_top() + DOWN * 0.3)
        
        # Agent data
        location = state.get('location', {'x': 0, 'z': 0})
        
        info_items = [
            ("Position", f"({location['x']:.1f}, {location['z']:.1f})"),
            ("Health", f"{state.get('hitpoint', 0)}"),
            ("Hunger", f"{state.get('hunger', 0.0):.1f}"),
            ("State", self._get_state_name(state.get('state', 0))),
            ("Frame", f"{frame_num}")
        ]
        
        info_content = VGroup()
        y_pos = 1.0
        
        for label, value in info_items:
            # Label
            label_text = Text(f"{label}:", font_size=14, color="#B0B0B0")
            # Value with color coding
            value_color = self._get_value_color(label, state)
            value_text = Text(value, font_size=14, color=value_color)
            
            # Arrange
            line_group = VGroup(label_text, value_text)
            line_group.arrange(RIGHT, buff=0.2)
            line_group.move_to(UP * y_pos)
            
            info_content.add(line_group)
            y_pos -= 0.4
        
        panel_content = VGroup(title, info_content)
        info_with_bg = VGroup(bg_panel, panel_content)
        info_with_bg.shift(LEFT * 4.5 + UP * 0.5)
        
        return info_with_bg
    
    def _get_state_name(self, state_num: int) -> str:
        """Convert state number to readable name"""
        state_names = {
            0: "Normal",
            1: "Fallen",
            2: "Victory",
            3: "Dead"
        }
        return state_names.get(state_num, f"State {state_num}")
    
    def _get_value_color(self, label: str, state: Dict) -> str:
        """Get color for values based on their meaning"""
        if label == "Health":
            hp = state.get('hitpoint', 100)
            if hp > 75: return "#4CAF50"  # Green
            elif hp > 25: return "#FF9800"  # Orange
            else: return "#F44336"  # Red
        elif label == "Hunger":
            hunger = state.get('hunger', 0)
            if hunger < 25: return "#4CAF50"  # Green
            elif hunger < 75: return "#FF9800"  # Orange
            else: return "#F44336"  # Red
        elif label == "State":
            state_num = state.get('state', 0)
            if state_num == 0: return "#4CAF50"  # Normal - green
            elif state_num == 1: return "#FF9800"  # Fallen - orange
            elif state_num == 2: return "#2196F3"  # Victory - blue
            else: return "#F44336"  # Dead - red
        else:
            return "#FFFFFF"  # White for other values
    
    def create_agent_with_direction(self) -> VGroup:
        """Create enhanced agent with direction indicator"""
        # Agent body with glow
        agent_glow = Circle(
            radius=0.25,
            color="#FFFFFF",
            fill_opacity=0.1,
            stroke_opacity=0.3,
            stroke_width=1
        )
        
        agent_body = Dot(
            radius=0.12, 
            color="#FFFFFF", 
            stroke_width=2,
            stroke_color="#E0E0E0",
            fill_opacity=0.9
        )
        
        # Direction arrow
        arrow = Arrow(
            start=ORIGIN, 
            end=UP*0.35, 
            color="#FFFFFF", 
            stroke_width=3,
            tip_length=0.12
        )
        
        # Direction label
        direction_label = Text("N", font_size=10, color="#FFFFFF").next_to(arrow, UP, buff=0.1)
        
        agent_group = VGroup(agent_glow, agent_body, arrow, direction_label)
        return agent_group
    
    def create_grid_background(self) -> VGroup:
        """Create an enhanced grid background"""
        grid = VGroup()
        
        # Grid lines
        for i in range(-4, 5):
            # Vertical lines
            v_line = Line(
                start=np.array([i, -3, 0]),
                end=np.array([i, 3, 0]),
                stroke_width=0.3,
                color="#2A2A2A",
                stroke_opacity=0.4
            )
            grid.add(v_line)
            
            # Horizontal lines
            h_line = Line(
                start=np.array([-4, i, 0]),
                end=np.array([4, i, 0]),
                stroke_width=0.3,
                color="#2A2A2A",
                stroke_opacity=0.4
            )
            grid.add(h_line)
        
        # Center cross (more prominent)
        center_v = Line(
            start=np.array([0, -3, 0]),
            end=np.array([0, 3, 0]),
            stroke_width=0.8,
            color="#404040",
            stroke_opacity=0.6
        )
        center_h = Line(
            start=np.array([-4, 0, 0]),
            end=np.array([4, 0, 0]),
            stroke_width=0.8,
            color="#404040",
            stroke_opacity=0.6
        )
        grid.add(center_v, center_h)
        
        return grid
    
    def construct(self):
        logger.info("Starting enhanced animation generation")
        
        # Load frame data
        self.load_frame_data()
        
        if not self.frames_data:
            logger.error("No frame data loaded")
            return
        
        if len(self.frames_data) < 2:
            logger.warning("Not enough frames for animation")
            return
        
        # Create background
        grid = self.create_grid_background()
        self.add(grid)
        
        # Create agent
        agent_group = self.create_agent_with_direction()
        
        # Create UI elements
        legend = self.create_enhanced_legend()
        
        # Add static elements
        self.add(agent_group, legend)
        
        # Enhanced title with subtitle
        title = Text(
            "SimsReal Agent View", 
            font_size=24,
            color="#FFFFFF"
        )
        subtitle = Text(
            "Real-time Ray Casting Visualization", 
            font_size=12,
            color="#B0B0B0"
        )
        title_group = VGroup(title, subtitle)
        title_group.arrange(DOWN, buff=0.1)
        title_group.to_edge(UP, buff=0.3)
        self.add(title_group)
        
        # Create all objects for all frames upfront
        all_dots = []
        all_lines = []
        all_infos = []
        
        for i, state in enumerate(self.frames_data):
            dots, lines = self.state_to_objects(state)
            info = self.create_info_panel(state, i * 3)  # Multiply by 3 since we sample every 3rd frame
            
            all_dots.append(dots)
            all_lines.append(lines)
            all_infos.append(info)
        
        # Show initial frame
        current_dots = all_dots[0]
        current_lines = all_lines[0]
        current_info = all_infos[0]
        
        for dot in current_dots:
            self.add(dot)
        for line in current_lines:
            self.add(line)
        self.add(current_info)
        
        # Animate each frame individually for smoother transitions
        for i in range(1, len(self.frames_data)):
            new_dots = all_dots[i]
            new_lines = all_lines[i]
            new_info = all_infos[i]
            
            frame_animations = []
            
            # Animate dots
            max_dots = max(len(current_dots), len(new_dots))
            for j in range(max_dots):
                if j < len(current_dots) and j < len(new_dots):
                    frame_animations.append(Transform(current_dots[j], new_dots[j]))
                elif j < len(new_dots):  # New object
                    frame_animations.append(FadeIn(new_dots[j]))
                    current_dots.append(new_dots[j])
                elif j < len(current_dots):  # Remove object
                    frame_animations.append(FadeOut(current_dots[j]))
            
            current_dots = current_dots[:len(new_dots)]
            
            # Animate lines
            max_lines = max(len(current_lines), len(new_lines))
            for j in range(max_lines):
                if j < len(current_lines) and j < len(new_lines):
                    frame_animations.append(Transform(current_lines[j], new_lines[j]))
                elif j < len(new_lines):
                    frame_animations.append(FadeIn(new_lines[j]))
                    current_lines.append(new_lines[j])
                elif j < len(current_lines):
                    frame_animations.append(FadeOut(current_lines[j]))
            
            current_lines = current_lines[:len(new_lines)]
            
            # Update info panel
            frame_animations.append(Transform(current_info, new_info))
            current_info = new_info
            
            # Play each frame individually with smooth timing
            if frame_animations:
                self.play(*frame_animations, run_time=0.1)  # 0.1 second per frame for smooth animation
            else:
                self.wait(0.1)
        
        # Hold final frame
        self.wait(1)
        self.play(FadeOut(*self.mobjects), run_time=1)


class AnimationGenerator:
    def __init__(self, snapshots_base_dir: str = "snapshots"):
        self.snapshots_base_dir = snapshots_base_dir
        self.output_dir = "animations"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_animation_for_run(self, run_number: int, quality: str = "medium_quality") -> Optional[str]:
        """Generate enhanced animation for a specific run with improved error handling"""
        run_dir = os.path.join(self.snapshots_base_dir, f"run{run_number}")
        
        if not os.path.exists(run_dir):
            logger.error(f"Run directory {run_dir} does not exist")
            return None
        
        # Check if there are enough frames
        json_files = glob.glob(os.path.join(run_dir, "frame_*.json"))
        if len(json_files) < 10:
            logger.warning(f"Not enough frames in {run_dir} (found {len(json_files)})")
            return None
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Use absolute paths to avoid file path issues
        output_file = os.path.abspath(os.path.join(self.output_dir, f"run{run_number}_enhanced.mp4"))
        
        try:
            # Clear any existing output file
            if os.path.exists(output_file):
                os.remove(output_file)
                logger.info(f"Removed existing output file: {output_file}")
            
            # Configure Manim with absolute paths
            config.quality = quality
            config.output_file = output_file
            config.background_color = "#0D1117"
            config.frame_rate = 20
            config.disable_caching = True
            
            # Ensure media directory exists
            media_dir = os.path.join(os.getcwd(), "media")
            os.makedirs(media_dir, exist_ok=True)
            
            # Create and render scene with error handling
            logger.info(f"Starting animation rendering for run {run_number}")
            scene = SimsRealAnimation(run_dir)
            scene.render()
            
            # Verify output file was created
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                logger.info(f"Enhanced animation saved to {output_file} (size: {file_size} bytes)")
                return output_file
            else:
                logger.error(f"Output file was not created: {output_file}")
                return None
            
        except Exception as e:
            logger.error(f"Unexpected error generating animation for run {run_number}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return None
    
    def generate_all_animations(self, quality: str = "medium_quality") -> List[str]:
        """Generate enhanced animations for all available runs"""
        if not os.path.exists(self.snapshots_base_dir):
            logger.error(f"Snapshots directory {self.snapshots_base_dir} does not exist")
            return []
        
        run_dirs = [d for d in os.listdir(self.snapshots_base_dir) 
                   if d.startswith("run") and os.path.isdir(os.path.join(self.snapshots_base_dir, d))]
        
        run_numbers = []
        for run_dir in run_dirs:
            try:
                run_num = int(run_dir[3:])
                run_numbers.append(run_num)
            except ValueError:
                continue
        
        run_numbers.sort()
        generated_files = []
        
        for run_num in run_numbers:
            logger.info(f"Generating enhanced animation for run {run_num}...")
            output_file = self.generate_animation_for_run(run_num, quality)
            if output_file:
                generated_files.append(output_file)
        
        logger.info(f"Generated {len(generated_files)} enhanced animations")
        return generated_files


if __name__ == "__main__":
    generator = AnimationGenerator()
    
    # Generate animation for latest run
    snapshots_dir = "snapshots"
    if os.path.exists(snapshots_dir):
        run_dirs = [d for d in os.listdir(snapshots_dir) if d.startswith("run")]
        if run_dirs:
            latest_run = max([int(d[3:]) for d in run_dirs])
            generator.generate_animation_for_run(latest_run)
        else:
            logger.error("No run directories found")
    else:
        logger.error("Snapshots directory not found") 