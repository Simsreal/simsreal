import json
import time
import zmq
import os
import argparse
from typing import Optional
from loguru import logger
import yaml
import torch
from simsreal_types import (
    State, Command, Action, DiscreteState, ActionType,
    state_to_discrete_state, discrete_state_to_hash, get_valid_actions,
    get_contextual_actions, ACTION_NAMES
)
from snapshot_manager import SnapshotManager
from intrinsics import IntrinsicSystem, create_intrinsic_system
from mcts import MCTS, create_mcts_agent, MCTSError


class ZMQRunner:
    def __init__(self, config: dict, debug_frames: bool = False, disable_snapshots: bool = False, disable_images: bool = False):
        self.cfg = config
        self.context = zmq.Context()
        self.first_message_saved = False
        
        # Initialize snapshot manager based on flags
        if disable_snapshots:
            self.snapshot_manager = None
            logger.info("Snapshot manager disabled via --disable-snapshots flag")
        else:
            self.snapshot_manager = SnapshotManager(enable_animations=False, debug_frames=debug_frames, disable_images=disable_images)
            if self.snapshot_manager:
                logger.info(f"Snapshot manager initialized: {self.snapshot_manager.get_current_run_info()}")
                if disable_images:
                    logger.info("Images disabled - only JSON metadata and exploration tracking will be saved")
        
        # GPU validation first
        self._validate_gpu_requirements()
        
        # Initialize MCTS agent (REQUIRED - no fallback allowed) with snapshot manager
        self.mcts_enabled = self.cfg.get('mcts', {}).get('enabled', True)
        self.fallback_to_simple = self.cfg.get('mcts', {}).get('fallback_to_simple', False)
        
        # Assert MCTS is enabled and no fallback is allowed
        assert self.mcts_enabled, "MCTS must be enabled - no fallback allowed"
        assert not self.fallback_to_simple, "Fallback to simple policy is not allowed"
        
        self.mcts_agent = None
        try:
            # Pass snapshot manager to MCTS for exploration tracking (None if disabled)
            self.mcts_agent = create_mcts_agent(config, self.snapshot_manager)
            if self.snapshot_manager:
                logger.info("MCTS agent initialized successfully with GPU acceleration and exploration tracking")
                logger.info("Fresh intrinsics will be created for each MCTS rollout")
                logger.info("Exploration data will be saved to snapshot directories")
            else:
                logger.info("MCTS agent initialized successfully with GPU acceleration (snapshot logging disabled)")
                logger.info("Fresh intrinsics will be created for each MCTS rollout (no exploration tracking)")
        except Exception as e:
            error_msg = f"CRITICAL: Failed to initialize MCTS agent: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # MCTS decision making parameters - FORCE MCTS EVERY FRAME
        original_frequency = self.cfg.get('mcts', {}).get('action_selection_frequency', 5)
        if original_frequency != 1:
            logger.warning(f"Config had action_selection_frequency={original_frequency}, forcing to 1 (MCTS every frame)")
        self.action_selection_frequency = 1  # ALWAYS use MCTS
        
        # Episode tracking (simplified - mainly for logging)
        self.current_episode = 0
        self.frame_count = 0
        self.mcts_decisions = 0
        self.mcts_failures = 0
        self.simple_calls_blocked = 0  # Track blocked simple calls
        
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(f"tcp://{self.cfg['robot']['sub']['ip']}:{self.cfg['robot']['sub']['port']}")
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        self.subscriber.setsockopt(zmq.RCVHWM, 10)
        self.subscriber.setsockopt(zmq.RCVTIMEO, 100)
        
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://{self.cfg['robot']['pub']['ip']}:{self.cfg['robot']['pub']['port']}")
        self.publisher.setsockopt(zmq.SNDHWM, 10)
        
        self.last_message_time = time.time()
        
        logger.info(f"ZMQ initialized: Sub {self.cfg['robot']['sub']['ip']}:{self.cfg['robot']['sub']['port']}, "
                   f"Pub {self.cfg['robot']['pub']['ip']}:{self.cfg['robot']['pub']['port']}")
        if self.snapshot_manager:
            logger.info(f"Snapshot manager info: {self.snapshot_manager.get_current_run_info()}")
        logger.info(f"Debug frames: {'ENABLED' if debug_frames else 'DISABLED'}")
        logger.info(f"MCTS: ENABLED (REQUIRED) with exploration tracking - EVERY FRAME")
        logger.info(f"Action selection: MCTS ONLY (frequency=1, simple action BLOCKED)")
        logger.info(f"Fallback policy: COMPLETELY DISABLED")
        
        # Final validation
        self._validate_initialization()

    def _validate_gpu_requirements(self):
        """Validate GPU requirements before initialization"""
        mcts_config = self.cfg.get('mcts', {})
        require_gpu = mcts_config.get('require_gpu', True)
        use_gpu = mcts_config.get('use_gpu', True)
        
        if require_gpu or use_gpu:
            if not torch.cuda.is_available():
                error_msg = "GPU is required but CUDA is not available!"
                logger.error(error_msg)
                logger.error("Please ensure:")
                logger.error("1. NVIDIA GPU is installed")
                logger.error("2. CUDA drivers are installed")
                logger.error("3. PyTorch with CUDA support is installed")
                raise RuntimeError(error_msg)
            
            # Test GPU memory
            try:
                device = torch.device('cuda')
                test_tensor = torch.randn(1000, 1000, device=device)
                del test_tensor
                torch.cuda.empty_cache()
                
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU validation passed: {gpu_name} ({gpu_memory:.1f}GB)")
                
            except Exception as e:
                error_msg = f"GPU validation failed: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

    def _validate_initialization(self):
        """Final validation of system initialization"""
        try:
            assert self.mcts_agent is not None, "MCTS agent not initialized"
            # Remove this assertion since snapshot manager can now be None:
            # assert self.snapshot_manager is not None, "Snapshot manager not initialized"
            assert self.action_selection_frequency == 1, "Action selection frequency must be 1 (MCTS every frame)"
            
            # Test MCTS agent with dummy state
            dummy_state = {
                "location": {"x": 0, "z": 0},
                "line_of_sight": [{"distance": 0.0, "type": 0}] * 72,
                "hitpoint": 100,
                "state": 0,
                "hunger": 50.0,
                "timestamp": int(time.time() * 1000),
                "snapshot": {
                    "data": [[0]], "width": 1, "height": 1, "resolution": 1.0,
                    "origin": {"x": 0, "z": 0}, "timestamp": int(time.time() * 1000)
                }
            }
            
            # Test MCTS search
            test_action = self.mcts_agent.search(dummy_state)
            assert test_action is not None, "MCTS test search failed"
            
            # Check exploration tracking integration (only if snapshot manager exists)
            if self.snapshot_manager:
                stats = self.mcts_agent.get_search_statistics()
                logger.info(f"Exploration tracking integration test: explorations_tracked={stats.get('explorations_tracked', 0)}")
            else:
                logger.info("Exploration tracking disabled - snapshot manager is None")
            
            logger.info("System initialization validation passed")
            logger.info("CONFIRMED: Only MCTS will be used for action selection")
            
        except Exception as e:
            error_msg = f"System validation failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _save_first_message(self, state_data: dict) -> None:
        if not self.first_message_saved:
            os.makedirs("payloads", exist_ok=True)
            with open("payloads/payload.json", "w") as f:
                json.dump(state_data, f, indent=2)
            logger.info("First message saved as payloads/payload.json")
            self.first_message_saved = True

    def _select_action_mcts(self, state: State) -> ActionType:
        """
        Use MCTS to select the best action (REQUIRED - no fallback)
        
        Args:
            state: Current game state
            
        Returns:
            Selected action
        """
        try:
            # Check if we're in a terminal state (won/dead) - wait for reset
            if state['state'] in [2, 3]:  # won or dead
                logger.debug(f"Agent in terminal state {state['state']}, waiting for episode reset...")
                # Return a safe default action (this won't be processed by Unity anyway)
                return ActionType.MOVE_FORWARD
            
            # Check if we have valid actions before proceeding
            valid_actions = get_contextual_actions(state['state'])
            if not valid_actions:
                logger.warning(f"No valid actions for agent state {state['state']}, using default action")
                # If agent is in fell_down state (1) but somehow no STANDUP action, return it anyway
                if state['state'] == 1:
                    return ActionType.STANDUP
                # Otherwise return safe default
                return ActionType.MOVE_FORWARD
            
            start_time = time.time()
            action = self.mcts_agent.search(state)
            search_time = time.time() - start_time
            
            self.mcts_decisions += 1
            
            logger.debug(f"MCTS selected action: {action.name} in {search_time:.3f}s")
            
            # Get action statistics and exploration info (reduced frequency)
            if self.frame_count % 500 == 0:  # Every 500 frames instead of 100
                action_probs = self.mcts_agent.get_action_probabilities(state)
                search_stats = self.mcts_agent.get_search_statistics()
                
                logger.info(f"Episode {self.current_episode}, Frame {self.frame_count} - "
                           f"MCTS action: {action.name}")
                logger.info(f"Action probabilities: {[(a.name, f'{p:.2f}') for a, p in action_probs.items()]}")
                logger.info(f"MCTS stats: iterations={search_stats['iterations_performed']}, "
                           f"nodes={search_stats['nodes_created']}, "
                           f"rollouts={search_stats['rollouts_performed']}, "
                           f"fresh_intrinsics={search_stats['fresh_intrinsics_created']}")
                logger.info(f"Exploration tracking: explorations={search_stats['explorations_tracked']}, "
                           f"tracking_enabled={search_stats['exploration_tracking_enabled']}")
                logger.info(f"GPU memory: {search_stats['gpu_memory_allocated']:.2f}GB allocated, "
                           f"{search_stats['gpu_memory_cached']:.2f}GB cached")
                logger.info(f"Simple calls blocked: {self.simple_calls_blocked} (should always be 0)")
                
                # Log exploration summary occasionally
                if self.frame_count % 2000 == 0 and self.snapshot_manager:
                    try:
                        exploration_summary = self.snapshot_manager.get_exploration_summary()
                        if "error" not in exploration_summary:
                            logger.info(f"Exploration summary: {exploration_summary['total_explorations']} total, "
                                       f"avg_reward={exploration_summary['avg_reward_per_exploration']:.3f}, "
                                       f"termination_reasons={exploration_summary['termination_reasons']}")
                    except Exception as e:
                        logger.warning(f"Failed to get exploration summary: {e}")
            
            return action
            
        except MCTSError as e:
            self.mcts_failures += 1
            error_msg = f"CRITICAL: MCTS action selection failed: {e}"
            logger.error(error_msg)
            # MCTS failure is critical - we cannot continue
            raise RuntimeError(error_msg)
        except Exception as e:
            self.mcts_failures += 1
            error_msg = f"CRITICAL: Unexpected error in MCTS action selection: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _select_action_simple_BLOCKED(self, state: State) -> ActionType:
        """
        BLOCKED: Simple action selection should NEVER be called
        This method will always raise an error to catch any bugs
        """
        self.simple_calls_blocked += 1
        error_msg = f"CRITICAL BUG: Simple action selection was called! This should NEVER happen."
        logger.error(error_msg)
        logger.error(f"Frame: {self.frame_count}, Episode: {self.current_episode}")
        logger.error(f"action_selection_frequency: {self.action_selection_frequency}")
        logger.error(f"This indicates a bug in the action selection logic!")
        raise RuntimeError(error_msg)
    
    def _select_action(self, state: State) -> ActionType:
        """
        Main action selection logic - ALWAYS AND ONLY use MCTS
        
        Args:
            state: Current game state
            
        Returns:
            Selected action
        """
        # ALWAYS use MCTS - NO frequency checking, NO fallback
        return self._select_action_mcts(state)

    def receive_state(self) -> Optional[State]:
        """
        Receive and normalize state data from simulation.
        Simplified - MCTS handles its own state processing.
        """
        try:
            if self.subscriber.poll(timeout=50):
                raw_message = self.subscriber.recv_string(zmq.NOBLOCK)
                state_data = json.loads(raw_message)
                self.last_message_time = time.time()
                
                # Save first message for debugging
                self._save_first_message(state_data)
                
                # Normalize state format
                normalized_state: State = self._normalize_state_format(state_data)
                
                # Optional: Save snapshot for debugging/visualization
                if self.snapshot_manager:
                    try:
                        self.snapshot_manager.save_snapshot(normalized_state)
                    except Exception as e:
                        logger.warning(f"Snapshot save failed: {e}")
                
                # Optional: Simple episode detection for logging (no complex reset logic)
                self._update_episode_counter(normalized_state['state'])
                
                # Clear message buffer
                messages_cleared = 0
                while self.subscriber.poll(timeout=0) and messages_cleared < 5:
                    try:
                        self.subscriber.recv_string(zmq.NOBLOCK)
                        messages_cleared += 1
                    except zmq.Again:
                        break
                
                return normalized_state
            else:
                current_time = time.time()
                if current_time - self.last_message_time > 5.0:
                    logger.warning("No messages received for 5 seconds")
                return None
        except zmq.Again:
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    def _update_episode_counter(self, current_agent_state: int):
        """Simple episode counter for logging purposes only"""
        if not hasattr(self, '_last_agent_state'):
            self._last_agent_state = current_agent_state
            return
        
        # Simple episode detection for logging
        if self._last_agent_state in [2, 3] and current_agent_state in [0, 1]:
            self.current_episode += 1
            self.frame_count = 0  # Reset frame counter for new episode
            logger.info(f"Episode {self.current_episode} started (agent state: {current_agent_state})")
            
            # Log exploration summary for completed episode
            if self.snapshot_manager and self.current_episode > 1:
                try:
                    prev_run = self.current_episode - 2  # Previous run number
                    exploration_summary = self.snapshot_manager.get_exploration_summary(prev_run)
                    if "error" not in exploration_summary:
                        logger.info(f"Episode {self.current_episode-1} exploration summary: "
                                   f"{exploration_summary['total_explorations']} explorations, "
                                   f"avg_reward={exploration_summary['avg_reward_per_exploration']:.3f}, "
                                   f"avg_steps={exploration_summary['avg_steps_per_exploration']:.1f}")
                except Exception as e:
                    logger.debug(f"Could not get exploration summary for previous episode: {e}")
            
            # Clear MCTS cache for new episode
            try:
                self.mcts_agent.clear_cache()
                logger.debug("MCTS cache cleared for new episode")
            except Exception as e:
                logger.warning(f"Failed to clear MCTS cache: {e}")
        
        self._last_agent_state = current_agent_state

    def _normalize_state_format(self, state_data: dict) -> State:
        """Normalize raw state data into standard State format"""
        location = {
            "x": int(state_data.get("x", 0)),
            "z": int(state_data.get("z", 0))
        }
        
        line_of_sight = []
        raw_los = state_data.get("line_of_sight", [])
        for item in raw_los:
            distance = item.get("Distance", item.get("distance", 0.0))
            obj_type = item.get("Type", item.get("type", 0))
            line_of_sight.append({
                "distance": float(distance),
                "type": int(obj_type)
            })
        
        # Generate snapshot from raycast data (for debugging/visualization)
        snapshot = {"data": [[0]], "width": 1, "height": 1, "resolution": 1.0,
                   "origin": {"x": 0, "z": 0}, "timestamp": int(time.time() * 1000)}
        
        if self.snapshot_manager:
            try:
                snapshot = self.snapshot_manager.generate_snapshot_from_raycast(location, line_of_sight)
            except Exception as e:
                logger.warning(f"Snapshot generation failed: {e}")
        
        return {
            "location": location, # type: ignore
            "line_of_sight": line_of_sight,
            "hitpoint": state_data.get("hit_point", state_data.get("hitpoint", 0)),
            "state": state_data.get("state", 0),
            "hunger": state_data.get("hunger", 0.0),
            "timestamp": state_data.get("timestamp", int(time.time() * 1000)),
            "snapshot": snapshot
        }

    def send_command(self, action: Action) -> None:
        try:
            command: Command = {
                "timestamp": int(time.time()),
                "action": action
            }
            self.publisher.send_string(json.dumps(command))
        except Exception as e:
            logger.error(f"Error sending command: {e}")

    def test_mcts_system(self, state: State) -> None:
        """Test method to demonstrate MCTS system capabilities with exploration tracking"""
        try:
            logger.info(f"Testing MCTS system for episode {self.current_episode}...")
            
            # Test MCTS action selection
            start_time = time.time()
            best_action = self.mcts_agent.search(state)
            search_time = time.time() - start_time
            
            # Get action probabilities
            action_probs = self.mcts_agent.get_action_probabilities(state)
            
            # Get search statistics
            search_stats = self.mcts_agent.get_search_statistics()
            
            logger.info(f"MCTS test results:")
            logger.info(f"  Best action: {best_action.name}")
            logger.info(f"  Search time: {search_time:.3f}s")
            logger.info(f"  Action probabilities:")
            for action, prob in action_probs.items():
                logger.info(f"    {action.name}: {prob:.3f}")
            logger.info(f"  Search statistics:")
            logger.info(f"    Iterations: {search_stats['iterations_performed']}")
            logger.info(f"    Nodes created: {search_stats['nodes_created']}")
            logger.info(f"    Rollouts: {search_stats['rollouts_performed']}")
            logger.info(f"    Fresh intrinsics: {search_stats['fresh_intrinsics_created']}")
            logger.info(f"    Explorations tracked: {search_stats['explorations_tracked']}")
            logger.info(f"    Average depth: {search_stats['average_depth_reached']:.1f}")
            logger.info(f"    GPU memory: {search_stats['gpu_memory_allocated']:.2f}GB allocated")
            
            # Test exploration summary
            if self.snapshot_manager:
                exploration_summary = self.snapshot_manager.get_exploration_summary()
                if "error" not in exploration_summary:
                    logger.info(f"  Exploration summary:")
                    logger.info(f"    Total explorations: {exploration_summary['total_explorations']}")
                    logger.info(f"    Avg reward: {exploration_summary['avg_reward_per_exploration']:.3f}")
                    logger.info(f"    Termination reasons: {exploration_summary['termination_reasons']}")
            
        except Exception as e:
            error_msg = f"MCTS test failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def run(self):
        logger.info("Starting ZMQ runner with MCTS integration and exploration tracking (GPU required)...")
        logger.info("MCTS handles all state processing internally with fresh intrinsics per rollout")
        logger.info("Exploration data will be saved for analysis")
        logger.info("USING MCTS FOR EVERY FRAME - NO SIMPLE FALLBACK EVER")
        
        try:
            while True:
                state = self.receive_state()
                if state:
                    self.frame_count += 1
                    
                    # Select action using MCTS ONLY - NO fallback logic
                    selected_action = self._select_action(state)
                    
                    # Convert to Unity command format
                    action: Action = {
                        "movement": ACTION_NAMES[selected_action],
                        "confidence": 0.8
                    }
                    
                    self.send_command(action)
                    
                    # Optional: Test MCTS system every 5000 frames
                    if self.frame_count % 5000 == 0:
                        logger.info("Running periodic MCTS system test...")
                        self.test_mcts_system(state)
                
                time.sleep(0.005)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            logger.info(f"Final episode: {self.current_episode}")
            logger.info(f"Total frames processed: {self.frame_count}")
            logger.info(f"MCTS decisions: {self.mcts_decisions}")
            logger.info(f"MCTS failures: {self.mcts_failures}")
            logger.info(f"Simple calls blocked: {self.simple_calls_blocked} (should be 0)")
            logger.info(f"MCTS usage rate: {100.0 * self.mcts_decisions / max(1, self.frame_count):.1f}% (should be 100%)")
            
            if self.mcts_agent:
                final_stats = self.mcts_agent.get_search_statistics()
                logger.info(f"Final MCTS stats: {final_stats}")
            
            if self.snapshot_manager:
                final_exploration_summary = self.snapshot_manager.get_exploration_summary()
                if "error" not in final_exploration_summary:
                    logger.info(f"Final exploration summary: {final_exploration_summary}")
                
        except RuntimeError as e:
            logger.error(f"CRITICAL ERROR: {e}")
            logger.error("System cannot continue - shutting down")
            raise
            
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up ZMQ connections...")
        try:
            if hasattr(self, "subscriber"):
                self.subscriber.close()
            if hasattr(self, "publisher"):
                self.publisher.close()
            if hasattr(self, "context"):
                self.context.term()
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='SimsReal ZMQ Runner with MCTS-ONLY action selection and exploration tracking (GPU required)')
    parser.add_argument('--debug-frames', action='store_true', 
                       help='Enable debug frame saving (saves every frame)')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--test-mcts', action='store_true',
                       help='Run MCTS system test on startup')
    parser.add_argument('--disable-snapshots', action='store_true',
                       help='Disable all snapshot/image logging for better performance')
    parser.add_argument('--disable-images', action='store_true',
                       help='Disable PNG image generation while keeping JSON metadata and exploration tracking')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Ensure MCTS is enabled and GPU is required
    mcts_config = config.setdefault('mcts', {})
    mcts_config['enabled'] = True
    mcts_config['require_gpu'] = True
    mcts_config['use_gpu'] = True
    mcts_config['fallback_to_simple'] = False
    mcts_config['enable_exploration_tracking'] = not args.disable_snapshots  # Disable exploration tracking if snapshots disabled
    mcts_config['action_selection_frequency'] = 1  # Force MCTS every frame
    
    try:
        runner = ZMQRunner(config, debug_frames=args.debug_frames, disable_snapshots=args.disable_snapshots, disable_images=args.disable_images)
        
        # Optional: Test MCTS system on startup
        if args.test_mcts:
            # Create a dummy state for testing
            dummy_state = {
                "location": {"x": 0, "z": 0},
                "line_of_sight": [
                    {"distance": 8.0, "type": 3},   # Trap nearby
                    {"distance": 15.0, "type": 2},  # Checkpoint
                    {"distance": 100.0, "type": 0}  # Empty space
                ] + [{"distance": 0.0, "type": 0}] * 69,  # Fill to 72 rays
                "hitpoint": 100,
                "state": 0,
                "hunger": 45.0,
                "timestamp": int(time.time() * 1000),
                "snapshot": {
                    "data": [[0]], "width": 1, "height": 1, "resolution": 1.0,
                    "origin": {"x": 0, "z": 0}, "timestamp": int(time.time() * 1000)
                }
            }
            runner.test_mcts_system(dummy_state)
        
        runner.run()
        
    except RuntimeError as e:
        logger.error(f"FATAL: {e}")
        logger.error("Cannot start system - please check GPU setup and configuration")
        exit(1)
    except Exception as e:
        logger.error(f"UNEXPECTED ERROR: {e}")
        exit(1)
