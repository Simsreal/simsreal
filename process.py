import json
import time
import torch
import torch.nn as nn
import zmq
import numpy as np
from typing import Dict, Any
from torchvision import transforms

from agi.ctxparser import ContextParser, VisionConstructor
from agi.memory_manager import MemoryManager
from agi.motivator import Motivator
from agi.governor import Governor
from pprint import pprint

from loguru import logger


class SequentialProcessor:
    def __init__(self, runtime_engine):
        self.runtime_engine = runtime_engine
        self.cfg = runtime_engine.get_metadata("config")
        self.device = runtime_engine.get_metadata("device")

        self.emb_dim = self.cfg["perceivers"]["vision"]["emb_dim"]

        self.skip_heavy_processing = False
        self.processing_counter = 0
        self.last_process_time = time.time()
        self.last_message_time = time.time()

        self._init_zmq()
        self._init_memory_manager()
        self._init_motivator()
        self._init_governor()
        self._init_context_parser()

    def _init_zmq(self):
        self.context = zmq.Context()

        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(
            f"tcp://{self.cfg['robot']['sub']['ip']}:{self.cfg['robot']['sub']['port']}"
        )
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b"")

        self.subscriber.setsockopt(zmq.RCVHWM, 10)

        self.subscriber.setsockopt(zmq.RCVTIMEO, 100)

        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(
            f"tcp://{self.cfg['robot']['pub']['ip']}:{self.cfg['robot']['pub']['port']}"
        )

        self.publisher.setsockopt(zmq.SNDHWM, 10)

        logger.info(
            "ZMQ connections initialized with balanced settings: "
            f"Subscriber at {self.cfg['robot']['sub']['ip']}:{self.cfg['robot']['sub']['port']}, "
            f"Publisher at {self.cfg['robot']['pub']['ip']}:{self.cfg['robot']['pub']['port']}"
        )

    def _init_memory_manager(self):
        self.memory_manager = MemoryManager(self.cfg, self.emb_dim)

        self.live_memory = self.memory_manager.live_memory
        self.episodic_memory = self.memory_manager.episodic_memory

    def _init_motivator(self):
        self.motivator = Motivator(
            self.cfg, self.device, self.emb_dim, self.live_memory, self.episodic_memory
        )

        self.intrinsics = self.motivator.intrinsics
        self.motivators = self.motivator.motivators
        self.intrinsic_indices = self.motivator.intrinsic_indices
        self.current_emotion_guidance = self.motivator.current_emotion_guidance

    def _init_governor(self):
        self.governor = Governor(self.cfg, self.device, self.emb_dim, self.intrinsics)

        self.movement_symbols = self.governor.movement_symbols
        self.titans_model = self.governor.titans_model
        self.titans_alphasr = self.governor.titans_alphasr
        self.governor_optimizer = self.governor.governor_optimizer
        self.ctx = self.governor.ctx
        self.governor_counter = self.governor.governor_counter

    def _init_context_parser(self):
        enable_viz = self.cfg.get("enable_mindmap_viz", True)
        save_frames = self.cfg.get("save_mindmap_frames", True)
        output_dir = self.cfg.get("mindmap_output_dir", "mindmap_frames")

        self.context_parser = ContextParser(
            device=self.device,
            enable_viz=enable_viz,
            save_frames=save_frames,
            output_dir=output_dir,
        )
        self.vision_constructor = VisionConstructor()

    def motivator_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.motivator.process_step(context)

    def ctx_parser(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.context_parser.parse_context(
            raw_data,
            construct_vision_fn=self.vision_constructor.construct_vision_from_raycast,
        )

    def memory_manager_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.memory_manager.process_step(context)

    def fifo_context_update(self, ctx, x):
        return self.governor.fifo_context_update(ctx, x)

    def governor_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return self.governor.process_step(context)

    def _tensor_to_serializable(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._tensor_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._tensor_to_serializable(item) for item in obj]
        else:
            return obj

    def process_step(self):
        """Process one step with improved message handling"""
        try:
            current_time = time.time()

            # 1. Try to receive a message with timeout
            try:
                # Use a longer timeout to ensure we can receive messages
                if self.subscriber.poll(timeout=50):  # 50ms timeout
                    raw_message = self.subscriber.recv_string(zmq.NOBLOCK)
                    raw_data = json.loads(raw_message)
                    pprint(raw_data)
                    self.last_message_time = current_time

                    # Optional: Clear additional messages if queue is building up
                    messages_cleared = 0
                    while self.subscriber.poll(timeout=0) and messages_cleared < 5:
                        try:
                            self.subscriber.recv_string(zmq.NOBLOCK)
                            messages_cleared += 1
                        except zmq.Again:
                            break

                    if messages_cleared > 0:
                        logger.debug(f"Cleared {messages_cleared} additional messages")

                else:
                    # No message available, but check if we haven't received anything for too long
                    if (
                        current_time - self.last_message_time > 5.0
                    ):  # 5 seconds without messages
                        logger.warning(
                            "No messages received for 5 seconds - check Unity connection"
                        )
                    return

            except zmq.Again:
                return  # No message available
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                return

            # 2. Adaptive processing based on timing
            processing_interval = current_time - self.last_process_time
            target_interval = 1.0 / self.cfg.get(
                "running_frequency", 20
            )  # 50ms for 20Hz

            # Skip heavy processing if we're falling behind
            self.skip_heavy_processing = processing_interval > target_interval * 1.5

            # 3. Parse context (lightweight)
            context = self.ctx_parser(raw_data)
            agent_state = context["agent_state"]
            if not context:
                return

            # 4. Conditional processing based on performance
            if not self.skip_heavy_processing:
                # Full processing when we have time
                memory_info = self.memory_manager_step(context)
                motivator_info = self.motivator_step(context)
            else:
                # Lightweight processing when behind
                memory_info = {}
                motivator_info = {"emotion_guidance": None}

            # 5. Always do governor step (critical for control)
            governor_info = self.governor_step(
                {
                    **context,
                    "emotion_guidance": motivator_info.get("emotion_guidance"),
                }
            )

            # 6. Send command immediately
            command_output = {
                "timestamp": int(current_time),
                "action": {
                    "movement": governor_info.get("movement_command", "idle"),
                    "confidence": governor_info.get("decision_confidence", 0.0),
                },
            }
            # pprint(command_output)

            self.publisher.send_string(json.dumps(command_output))
            self.last_process_time = current_time
            self.processing_counter += 1

            # Debug info
            if self.processing_counter % 100 == 0:
                logger.info(
                    f"Processing: {'FAST' if self.skip_heavy_processing else 'FULL'} mode, "
                    f"interval: {processing_interval:.3f}s, msgs/sec: {100/(current_time - self.last_message_time + 0.001):.1f}"
                )

        except Exception as e:
            logger.error(f"Process step error: {e}")

    def run(self):
        logger.info("Starting processing loop...")
        try:
            while True:
                self.process_step()
                time.sleep(0.005)

        except KeyboardInterrupt:
            logger.info("Shutting down processor...")
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up ZMQ connections...")

        if hasattr(self, "subscriber"):
            self.subscriber.close()
        if hasattr(self, "publisher"):
            self.publisher.close()

        if hasattr(self, "context"):
            self.context.term()

        logger.info("Cleanup completed")
