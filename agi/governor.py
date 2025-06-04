# import time
# from enum import IntEnum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from agi.learning.conscious.titans import Titans
from agi.learning.emotions import TitansAlphaSR
from src.utilities.emotion.pad import (
    get_emotion_magnitude,
    get_emotion_reward,
)
from src.utilities.queues.queue_util import try_get


def governor(runtime_engine):
    def fifo(ctx, x) -> torch.Tensor:
        x = x.to(device)
        return torch.cat((ctx[:, 1:, :], x.unsqueeze(1)), dim=1)

    cfg = runtime_engine.get_metadata("config")
    device = runtime_engine.get_metadata("device")
    latent_dim = runtime_engine.get_metadata("latent_size")
    intrinsics = cfg["intrinsics"]
    
    movement_symbols = ["moveup", "movedown", "moveleft", "moveright", "idle", "standup"]
    movement_dim = len(movement_symbols)
    intrinsic_dim = len(intrinsics)
    total_policy_dim = intrinsic_dim + movement_dim
    
    brain_shm = runtime_engine.get_shared_memory("brain")
    actuator_shm = runtime_engine.get_shared_memory("actuator")
    motivator_shm = runtime_engine.get_shared_memory("motivator")

    brain_cfg = cfg["brain"]
    ctx_len = brain_cfg["ctx_len"]

    # Initialize Titans
    titans_model = Titans(
        latent_dim,
        brain_cfg["titans"]["chunk_size"],
        device,
        total_policy_dim,
    ).to(device)

    titans_alphasr = TitansAlphaSR(titans_model, intrinsics, movement_symbols, device)
    optimizer = torch.optim.Adam(titans_model.parameters(), lr=0.001)
    counter = 0
    stream = torch.cuda.Stream(device=device)
    ctx = torch.zeros((1, ctx_len, latent_dim), dtype=torch.float32).to(device)

    logger.info("Governor configured with integrated emotion handling")

    while True:
        with torch.cuda.stream(stream): # type: ignore
            latent = try_get(brain_shm["latent"], device)
            emotion_guidance = try_get(motivator_shm["emotion_guidance"], device)
            
            if latent is None:
                continue

            ctx = fifo(ctx, latent)
            state = latent.flatten()
            
            # Get outputs from TitansAlphaSR with emotion guidance
            outputs = titans_alphasr.forward(
                state, 
                ctx, 
                emotion_guidance=emotion_guidance,
                optimizer=optimizer
            )
            
            # Convert to symbolic movement command
            movement_logits = outputs['movement_logits']
            movement_command_idx = int(torch.argmax(movement_logits, dim=-1).item())
            movement_command = movement_symbols[movement_command_idx]
            
            # Send movement command
            actuator_shm["torque"].put(movement_command)
            
            logger.debug(f"Movement: {movement_command}, Reward: {outputs['reward']:.3f}")

            # MCTS maintenance
            if counter % cfg.get("mcts", {}).get("decay_period", 6000) == 0:
                titans_alphasr.decay_visits()

            if counter % cfg.get("mcts", {}).get("prune_period", 6000) == 0:
                titans_alphasr.prune_states()

            counter += 1
