#!/usr/bin/env python3
"""
Rollout Results Analyzer (Optimized with Sampling)

This script analyzes rollout summary results from simulation runs 80-260.
It samples every 20th exploration for efficiency while maintaining statistical validity.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import statistics
from datetime import datetime

@dataclass
class ExplorationStats:
    """Statistics for a single exploration"""
    exploration_id: str
    total_reward: float
    total_steps: int
    termination_reason: str
    avg_reward_per_step: float
    completed_timestamp: int
    rollout_id: str
    mcts_iteration: int
    intrinsic_breakdown: Dict[str, float]

@dataclass
class RunSummary:
    """Summary statistics for an entire run"""
    run_id: int
    total_explorations: int
    sampled_explorations: int  # How many we actually analyzed
    total_reward: float
    avg_reward: float
    max_reward: float
    min_reward: float
    total_steps: int
    avg_steps: float
    max_steps: int
    min_steps: int
    termination_reasons: Dict[str, int]
    intrinsic_stats: Dict[str, Dict[str, float]]
    timestamp_range: Tuple[int, int]
    mcts_iterations: Dict[int, int]

class RolloutAnalyzer:
    """Main analyzer class for processing rollout results"""
    
    def __init__(self, snapshots_dir: str = "snapshots", sample_rate: int = 20):
        self.snapshots_dir = Path(snapshots_dir)
        self.sample_rate = sample_rate  # Sample every Nth exploration
        self.run_summaries: Dict[int, RunSummary] = {}
    
    def load_exploration_data(self, run_id: int) -> Tuple[List[ExplorationStats], int]:
        """Load sampled exploration data for a specific run"""
        explorations_dir = self.snapshots_dir / f"run{run_id}" / "explorations"
        
        if not explorations_dir.exists():
            print(f"Warning: Run {run_id} explorations directory not found")
            return [], 0
        
        exploration_dirs = sorted([d for d in explorations_dir.iterdir() if d.is_dir()])
        total_explorations = len(exploration_dirs)
        
        # Sample every Nth exploration
        sampled_dirs = exploration_dirs[::self.sample_rate]
        sampled_count = len(sampled_dirs)
        
        print(f"Processing run {run_id}: {total_explorations} total, sampling {sampled_count} (every {self.sample_rate}th)...")
        
        explorations = []
        for exploration_dir in sampled_dirs:
            rollout_file = exploration_dir / "rollout_summary.json"
            
            if not rollout_file.exists():
                continue
            
            try:
                with open(rollout_file, 'r') as f:
                    data = json.load(f)
                
                exploration = ExplorationStats(
                    exploration_id=data.get("exploration_id", ""),
                    total_reward=data.get("total_reward", 0.0),
                    total_steps=data.get("total_steps", 0),
                    termination_reason=data.get("termination_reason", "unknown"),
                    avg_reward_per_step=data.get("avg_reward_per_step", 0.0),
                    completed_timestamp=data.get("completed_timestamp", 0),
                    rollout_id=data.get("rollout_id", ""),
                    mcts_iteration=data.get("mcts_iteration", 0),
                    intrinsic_breakdown=data.get("intrinsic_breakdown", {})
                )
                explorations.append(exploration)
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing {rollout_file}: {e}")
        
        return explorations, total_explorations
    
    def analyze_run(self, run_id: int) -> Optional[RunSummary]:
        """Analyze a single run and return summary statistics"""
        explorations, total_explorations = self.load_exploration_data(run_id)
        
        if not explorations:
            return None
        
        sampled_count = len(explorations)
        
        # Basic statistics (scaled up from sample)
        scale_factor = total_explorations / sampled_count if sampled_count > 0 else 1
        
        rewards = [e.total_reward for e in explorations]
        steps = [e.total_steps for e in explorations]
        timestamps = [e.completed_timestamp for e in explorations if e.completed_timestamp > 0]
        
        # Termination reasons (scaled)
        termination_reasons = defaultdict(int)
        for e in explorations:
            termination_reasons[e.termination_reason] += 1
        
        # Scale up the counts
        for reason in termination_reasons:
            termination_reasons[reason] = int(termination_reasons[reason] * scale_factor)
        
        # MCTS iterations (scaled)
        mcts_iterations = defaultdict(int)
        for e in explorations:
            mcts_iterations[e.mcts_iteration] += 1
        
        for iteration in mcts_iterations:
            mcts_iterations[iteration] = int(mcts_iterations[iteration] * scale_factor)
        
        # Intrinsic breakdown statistics
        intrinsic_stats = defaultdict(lambda: {'values': []})
        for e in explorations:
            for metric, value in e.intrinsic_breakdown.items():
                intrinsic_stats[metric]['values'].append(value)
        
        # Calculate intrinsic statistics
        for metric in intrinsic_stats:
            values = intrinsic_stats[metric]['values']
            if values:
                intrinsic_stats[metric] = {
                    'avg': statistics.mean(values),
                    'max': max(values),
                    'min': min(values),
                    'total': sum(values) * scale_factor,  # Scale up total
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0
                }
        
        return RunSummary(
            run_id=run_id,
            total_explorations=total_explorations,
            sampled_explorations=sampled_count,
            total_reward=sum(rewards) * scale_factor,  # Scale up total reward
            avg_reward=statistics.mean(rewards) if rewards else 0.0,
            max_reward=max(rewards) if rewards else 0.0,
            min_reward=min(rewards) if rewards else 0.0,
            total_steps=sum(steps) * scale_factor,  # Scale up total steps
            avg_steps=statistics.mean(steps) if steps else 0.0,
            max_steps=max(steps) if steps else 0,
            min_steps=min(steps) if steps else 0,
            termination_reasons=dict(termination_reasons),
            intrinsic_stats=dict(intrinsic_stats),
            timestamp_range=(min(timestamps) if timestamps else 0, 
                           max(timestamps) if timestamps else 0),
            mcts_iterations=dict(mcts_iterations)
        )
    
    def analyze_range(self, start_run: int, end_run: int) -> Dict[int, RunSummary]:
        """Analyze a range of runs"""
        print(f"Analyzing runs {start_run} to {end_run} (sampling every {self.sample_rate}th exploration)...")
        
        summaries = {}
        for run_id in range(start_run, end_run + 1):
            summary = self.analyze_run(run_id)
            if summary:
                summaries[run_id] = summary
                print(f"✓ Run {run_id}: {summary.total_explorations} total ({summary.sampled_explorations} sampled)")
            else:
                print(f"✗ Run {run_id}: No data found")
        
        return summaries
    
    def print_summary_report(self, summaries: Dict[int, RunSummary]):
        """Print a comprehensive summary report"""
        print("\n" + "="*80)
        print("ROLLOUT ANALYSIS SUMMARY REPORT (SAMPLED)")
        print("="*80)
        
        if not summaries:
            print("No data to analyze!")
            return
        
        # Overall statistics
        total_runs = len(summaries)
        total_explorations = sum(s.total_explorations for s in summaries.values())
        total_sampled = sum(s.sampled_explorations for s in summaries.values())
        total_rewards = sum(s.total_reward for s in summaries.values())
        total_steps = sum(s.total_steps for s in summaries.values())
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Runs analyzed: {total_runs}")
        print(f"  Total explorations: {total_explorations:,}")
        print(f"  Total sampled: {total_sampled:,} (sampling rate: 1/{self.sample_rate})")
        print(f"  Estimated total rewards: {total_rewards:.2f}")
        print(f"  Estimated total steps: {total_steps:,}")
        print(f"  Average explorations per run: {total_explorations/total_runs:.1f}")
        
        # Per-run summary table
        print(f"\nPER-RUN SUMMARY:")
        print(f"{'Run':>4} {'Total Expl':>10} {'Sampled':>8} {'Avg Reward':>12} {'Max Reward':>12} {'Avg Steps':>10} {'Max Steps':>10}")
        print("-" * 80)
        
        for run_id in sorted(summaries.keys()):
            s = summaries[run_id]
            print(f"{run_id:>4} {s.total_explorations:>10,} {s.sampled_explorations:>8} {s.avg_reward:>12.3f} {s.max_reward:>12.3f} "
                  f"{s.avg_steps:>10.1f} {s.max_steps:>10}")
        
        # Termination reasons across all runs
        print(f"\nTERMINATION REASONS ACROSS ALL RUNS (ESTIMATED):")
        all_terminations = defaultdict(int)
        for s in summaries.values():
            for reason, count in s.termination_reasons.items():
                all_terminations[reason] += count
        
        for reason, count in sorted(all_terminations.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_explorations) * 100
            print(f"  {reason}: {count:,} ({percentage:.1f}%)")
        
        # Intrinsic metrics summary
        print(f"\nINTRINSIC METRICS SUMMARY:")
        all_intrinsic = defaultdict(list)
        for s in summaries.values():
            for metric, stats in s.intrinsic_stats.items():
                all_intrinsic[metric].append(stats['avg'])
        
        for metric in sorted(all_intrinsic.keys()):
            values = all_intrinsic[metric]
            if values:
                avg_across_runs = statistics.mean(values)
                max_across_runs = max(values)
                print(f"  {metric}: avg={avg_across_runs:.4f}, max={max_across_runs:.4f}")
    
    def save_detailed_report(self, summaries: Dict[int, RunSummary], filename: str = "rollout_analysis_report_sampled.json"):
        """Save detailed analysis to JSON file"""
        report_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "sample_rate": self.sample_rate,
            "summary_count": len(summaries),
            "runs": {}
        }
        
        for run_id, summary in summaries.items():
            report_data["runs"][str(run_id)] = {
                "run_id": summary.run_id,
                "total_explorations": summary.total_explorations,
                "sampled_explorations": summary.sampled_explorations,
                "total_reward": summary.total_reward,
                "avg_reward": summary.avg_reward,
                "max_reward": summary.max_reward,
                "min_reward": summary.min_reward,
                "total_steps": summary.total_steps,
                "avg_steps": summary.avg_steps,
                "max_steps": summary.max_steps,
                "min_steps": summary.min_steps,
                "termination_reasons": summary.termination_reasons,
                "intrinsic_stats": summary.intrinsic_stats,
                "timestamp_range": summary.timestamp_range,
                "mcts_iterations": summary.mcts_iterations
            }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nDetailed report saved to: {filename}")

def main():
    """Main function to run the analysis"""
    print("Rollout Results Analyzer (Optimized)")
    print("=" * 40)
    
    # Use sampling every 20th exploration for speed
    analyzer = RolloutAnalyzer(sample_rate=20)
    
    # Analyze runs 80-260
    summaries = analyzer.analyze_range(80, 260)
    
    # Print summary report
    analyzer.print_summary_report(summaries)
    
    # Save detailed report
    analyzer.save_detailed_report(summaries)
    
    print(f"\nAnalysis complete! Processed {len(summaries)} runs with sampling.")

if __name__ == "__main__":
    main() 