import networkx as nx
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path


def find_shm_patterns(file_path):
    """Find shared memory patterns in a file."""
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Pattern to match: something_shm["key"].put, including f-strings
        pattern = r'(\w+)_shm\[(?:f?["\']|f"|\')([^"\']+)["\']\]\.put'
        matches = re.findall(pattern, content)

        # Debug output
        if matches:
            print(f"\nDebug - Found in {file_path}:")
            for target, key in matches:
                print(f"  {target}_shm[{key}].put")

        return matches
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def get_process_name(file_path):
    """Convert file path to process name."""
    # Convert path like 'agi/ctx_parser.py' to 'agi.ctx_parser'
    parts = file_path.parts
    if "agi" in parts:
        agi_index = parts.index("agi")
        relevant_parts = parts[agi_index:]
        process_name = ".".join(relevant_parts)
        return os.path.splitext(process_name)[0]
    return None


def get_custom_layout():
    """Define custom positions for each node to match kawaii-style layout."""
    positions = {
        # Control nodes (top)
        "agi.motivator": (0.25, 0.85),  # Top-left control
        "agi.governor": (0.75, 0.85),  # Top-right control
        # Core processing (middle)
        "agi.brain": (0.5, 0.6),  # Center
        "agi.memory_manager": (0.25, 0.6),  # Left-center
        "agi.actuator": (0.75, 0.6),  # Right-center
        # Input layer (bottom)
        "agi.perceiver": (0.5, 0.35),  # Bottom-center
        "agi.ctx_parser": (0.75, 0.35),  # Bottom-right
        # External
        "Unity": (0.95, 0.35),  # Far right, aligned with ctx_parser
    }
    return positions


def get_node_colors():
    """Define colors and themes for nodes."""
    return {
        # Input/Processing nodes - Blue theme
        "agi.perceiver": {
            "color": "#5DADE2",  # Medium blue
            "edge_color": "#AED6F1",  # Light blue
        },
        "agi.ctx_parser": {
            "color": "#2E86C1",  # Darker blue
            "edge_color": "#85C1E9",  # Blue
        },
        # Core nodes - Red theme
        "agi.brain": {
            "color": "#EC7063",  # Medium red
            "edge_color": "#F5B7B1",  # Light red
        },
        # Memory node - Purple theme
        "agi.memory_manager": {
            "color": "#A569BD",  # Medium purple
            "edge_color": "#D2B4DE",  # Light purple
        },
        # Control nodes - Yellow theme
        "agi.governor": {
            "color": "#F4D03F",  # Medium yellow
            "edge_color": "#F9E79F",  # Light yellow
        },
        "agi.motivator": {
            "color": "#F1C40F",  # Darker yellow
            "edge_color": "#F7DC6F",  # Yellow
        },
        # Output node - Green theme
        "agi.actuator": {
            "color": "#45B39D",  # Medium green
            "edge_color": "#A2D9CE",  # Light green
        },
        # External node - Gray theme
        "Unity": {
            "color": "#95A5A6",  # Medium gray
            "edge_color": "#D5D8DC",  # Light gray
        },
    }


def get_edge_colors():
    """Define edge colors based on data type."""
    return {
        "vision": "#85C1E9",  # Blue - sensory data
        "latent": "#C39BD3",  # Purple - processed features
        "emotion": "#F1948A",  # Pink - emotional state
        "torque": "#7DCEA0",  # Green - motor commands
        "force_on_geoms": "#7DCEA0",  # Green - physics data
        "jnt_state": "#7DCEA0",  # Green - physics data
        "guidance": "#F4D03F",  # Yellow - control signals
        "governance": "#F4D03F",  # Yellow - control signals
        "commands": "#95A5A6",  # Gray - Unity interface
        "states": "#95A5A6",  # Gray - Unity interface
    }


def generate_network_relationship_graph():
    processes = [
        "agi.ctx_parser",
        "agi.perceiver",
        "agi.memory_manager",
        "agi.brain",
        "agi.governor",
        "agi.motivator",
        "agi.actuator",
        "Unity",
    ]
    G = nx.MultiDiGraph()

    # -------- nodes --------
    node_colors = get_node_colors()
    for process in processes:
        G.add_node(process, **node_colors[process])

    # -------- edges --------
    # Find all Python files in the agi directory
    agi_dir = Path("agi/")
    python_files = list(agi_dir.rglob("*.py"))

    # Track edges and their labels
    edge_labels = {}

    # Analyze each file for shared memory patterns
    print("Known processes:", processes)
    print("\nAnalyzing files for shared memory patterns...")

    for file_path in python_files:
        source_process = get_process_name(file_path)
        if not source_process or source_process not in processes:
            continue

        patterns = find_shm_patterns(file_path)
        if patterns:
            print(f"\nIn {source_process}:")
            for target, key in patterns:
                target_process = f"agi.{target}"
                if target_process in processes:
                    # Add edge with key as data
                    G.add_edge(source_process, target_process, key=key)
                    # Track edge label
                    edge_key = (source_process, target_process)
                    if edge_key not in edge_labels:
                        edge_labels[edge_key] = []
                    edge_labels[edge_key].append(key)

    # Add Unity I/O edges manually with specific colors
    edge_colors = get_edge_colors()
    G.add_edge("agi.actuator", "Unity", key="commands", color=edge_colors["commands"])
    G.add_edge("Unity", "agi.ctx_parser", key="states", color=edge_colors["states"])
    edge_labels[("agi.actuator", "Unity")] = ["commands"]
    edge_labels[("Unity", "agi.ctx_parser")] = ["states"]

    # Add motivator -> brain edge manually
    G.add_edge(
        "agi.motivator", "agi.brain", key="guidance", color=edge_colors["guidance"]
    )
    if ("agi.motivator", "agi.brain") not in edge_labels:
        edge_labels[("agi.motivator", "agi.brain")] = []
    edge_labels[("agi.motivator", "agi.brain")].append("guidance")

    # Create larger plot with white background
    plt.figure(figsize=(20, 16), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    # Use spring layout with tuned parameters
    pos = nx.spring_layout(
        G,
        k=4.0,  # More space between nodes
        iterations=150,  # More iterations for better convergence
        seed=90,  # Fixed seed for reproducibility
    )

    # Draw edges with node-specific colors
    edge_colors_list = []
    for u, v, k in G.edges(data=True):
        edge_colors_list.append(G.nodes[u]["edge_color"])

    # Draw with adjusted parameters
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=[G.nodes[node]["color"] for node in G.nodes()],
        edge_color=edge_colors_list,
        node_size=[
            7000 if node == "Unity" else 2500 for node in G.nodes()
        ],  # Bigger Unity node
        font_size=14,
        font_weight="bold",
        font_family="Source Code Pro",
        arrows=True,
        arrowsize=30,
        connectionstyle="arc3,rad=0.2",
        width=2.5,
    )

    # Draw edge labels with matching colors
    for (src, tgt), keys in edge_labels.items():
        label = "\n".join(keys[:2])
        if len(keys) > 2:
            label += "\n..."

        # Get source node's edge color for the label
        label_color = G.nodes[src]["edge_color"]

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={(src, tgt): label},
            font_size=14,
            font_family="Source Code Pro",  # Add font family for edge labels
            label_pos=0.6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=0.5),
            font_color=label_color,
            rotate=False,
        )

    # Save with higher quality and padding
    plt.savefig(
        "process_graph.png",
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
        edgecolor="none",
        pad_inches=0.3,
    )  # More padding around the plot
    plt.close()

    print("\nGraph saved as process_graph.png")


if __name__ == "__main__":
    generate_network_relationship_graph()
