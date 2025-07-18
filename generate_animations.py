#!/usr/bin/env python3
"""
Enhanced CLI tool to generate Manim animations from SimsReal snapshots
"""
import argparse
import sys
import os
from animation_generator import AnimationGenerator
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced Manim animations from SimsReal snapshots")
    parser.add_argument("--run", type=int, help="Generate animation for specific run number")
    parser.add_argument("--all", action="store_true", help="Generate animations for all runs")
    parser.add_argument("--latest", action="store_true", help="Generate animation for latest run")
    parser.add_argument("--quality", choices=["low_quality", "medium_quality", "high_quality"], 
                       default="medium_quality", help="Animation quality")
    parser.add_argument("--snapshots-dir", default="snapshots", help="Snapshots directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    if not any([args.run is not None, args.all, args.latest]):
        logger.error("Must specify either --run <number>, --all, or --latest")
        sys.exit(1)
    
    # Check if snapshots directory exists
    if not os.path.exists(args.snapshots_dir):
        logger.error(f"Snapshots directory does not exist: {args.snapshots_dir}")
        sys.exit(1)
    
    generator = AnimationGenerator(args.snapshots_dir)
    
    if args.run is not None:
        logger.info(f"Generating animation for run {args.run}")
        output = generator.generate_animation_for_run(args.run, args.quality)
        if output:
            print(f"Animation saved: {output}")
        else:
            logger.error("Animation generation failed")
            sys.exit(1)
    
    elif args.latest:
        # Find latest run
        run_dirs = [d for d in os.listdir(args.snapshots_dir) if d.startswith("run")]
        if not run_dirs:
            logger.error("No run directories found")
            sys.exit(1)
        
        latest_run = max([int(d[3:]) for d in run_dirs])
        logger.info(f"Generating animation for latest run {latest_run}")
        output = generator.generate_animation_for_run(latest_run, args.quality)
        if output:
            print(f"Animation saved: {output}")
        else:
            logger.error("Animation generation failed")
            sys.exit(1)
    
    elif args.all:
        logger.info("Generating animations for all runs")
        outputs = generator.generate_all_animations(args.quality)
        print(f"Generated {len(outputs)} animations:")
        for output in outputs:
            print(f"  {output}")


if __name__ == "__main__":
    main() 