import gc
import json
import os
import argparse
from typing import Any, Dict

import yaml
from loguru import logger
import torch

from run import ZMQRunner
from src.utilities.tools.retry import retry


class RuntimeEngine:
    def __init__(self):
        self.shared_data: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def add_shared_data(self, name: str, data: Any):
        self.shared_data[name] = data

    def get_shared_data(self, name: str) -> Any:
        return self.shared_data.get(name, {})

    def add_metadata(self, name: str, data: Any):
        self.metadata[name] = data

    def get_metadata(self, name: str) -> Any:
        return self.metadata.get(name)


def initialize_runtime_engine():
    runtime_engine = RuntimeEngine()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runtime_engine.add_metadata("config", config)
    runtime_engine.add_metadata("device", device)

    logger.info(f"Runtime engine initialized with device: {device}")
    return runtime_engine


def parse_arguments():
    parser = argparse.ArgumentParser(description='SimsReal Main Runtime Engine')
    parser.add_argument('--debug-frames', action='store_true', 
                       help='Enable debug frame saving (saves every frame)')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--exploration-summary', type=int, metavar='RUN_NUMBER',
                       help='Show exploration summary for specific run number (or latest if no number given)')
    parser.add_argument('--list-runs', action='store_true',
                       help='List all available runs')
    parser.add_argument('--disable-images', action='store_true',
                       help='Disable PNG image generation while keeping JSON metadata and exploration tracking')
    return parser.parse_args()


def show_exploration_summary(run_number: int = None):
    """Show exploration summary for a specific run"""
    from snapshot_manager import SnapshotManager
    
    # Create snapshot manager to access exploration data
    snapshot_manager = SnapshotManager()
    
    # Get exploration summary
    summary = snapshot_manager.get_exploration_summary(run_number)
    
    if "error" in summary:
        logger.error(summary["error"])
        return
    
    # Print formatted summary
    print(f"\n{'='*60}")
    print(f"EXPLORATION SUMMARY FOR RUN {summary['run_number']}")
    print(f"{'='*60}")
    print(f"Total Explorations: {summary['total_explorations']}")
    print(f"Total Reward: {summary['total_reward']:.3f}")
    print(f"Total Steps: {summary['total_steps']}")
    print(f"Avg Reward/Exploration: {summary['avg_reward_per_exploration']:.3f}")
    print(f"Avg Steps/Exploration: {summary['avg_steps_per_exploration']:.1f}")
    
    print(f"\nTermination Reasons:")
    for reason, count in summary['termination_reasons'].items():
        percentage = (count / summary['total_explorations']) * 100
        print(f"  {reason}: {count} ({percentage:.1f}%)")
    
    print(f"\nTop 10 Best Explorations (by reward):")
    top_explorations = sorted(summary['explorations'], 
                            key=lambda x: x.get('total_reward', 0), 
                            reverse=True)[:10]
    
    for i, exp in enumerate(top_explorations, 1):
        print(f"  {i:2d}. ID:{exp['exploration_id']} "
              f"Reward:{exp['total_reward']:7.3f} "
              f"Steps:{exp['total_steps']:3d} "
              f"Reason:{exp['termination_reason']}")
    
    print(f"{'='*60}\n")


def list_available_runs():
    """List all available runs"""
    snapshots_dir = "snapshots"
    if not os.path.exists(snapshots_dir):
        logger.error("No snapshots directory found")
        return
    
    run_dirs = []
    for item in os.listdir(snapshots_dir):
        if item.startswith("run") and os.path.isdir(os.path.join(snapshots_dir, item)):
            try:
                run_num = int(item[3:])
                run_path = os.path.join(snapshots_dir, item)
                
                # Count frames and explorations
                frame_count = len([f for f in os.listdir(run_path) 
                                 if f.endswith('.png') and f.startswith('frame_')])
                
                explorations_path = os.path.join(run_path, 'explorations')
                exploration_count = 0
                if os.path.exists(explorations_path):
                    exploration_count = len([d for d in os.listdir(explorations_path) 
                                           if os.path.isdir(os.path.join(explorations_path, d))])
                
                run_dirs.append((run_num, frame_count, exploration_count))
            except ValueError:
                continue
    
    if not run_dirs:
        logger.info("No runs found")
        return
    
    print(f"\n{'='*50}")
    print(f"AVAILABLE RUNS")
    print(f"{'='*50}")
    print(f"{'Run':>3} {'Frames':>7} {'Explorations':>12}")
    print(f"{'-'*50}")
    
    for run_num, frame_count, exploration_count in sorted(run_dirs):
        print(f"{run_num:3d} {frame_count:7d} {exploration_count:12d}")
    
    print(f"{'='*50}\n")


def main():
    try:
        args = parse_arguments()
        
        # Handle special commands that don't require full runtime initialization
        if args.list_runs:
            list_available_runs()
            return
        
        if args.exploration_summary is not None:
            show_exploration_summary(args.exploration_summary)
            return
        
        # Normal runtime initialization
        runtime_engine = initialize_runtime_engine()
        config = runtime_engine.get_metadata("config")
        
        runner = ZMQRunner(config, debug_frames=args.debug_frames, disable_images=getattr(args, 'disable_images', False))
        runner.run()

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
