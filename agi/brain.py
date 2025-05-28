# import time

import torch
import torch.nn.functional as F

from loguru import logger

from agi.learning.conscious.titans import Titans
from agi.learning.conscious.lstm import LSTM
from src.utilities.queues.queue_util import try_get


def brain(runtime_engine):
    def fifo(ctx, x) -> torch.Tensor:
        x = x.to(device)
        return torch.cat((ctx[:, 1:, :], x.unsqueeze(1)), dim=1)

    cfg = runtime_engine.get_metadata("config")
    device = runtime_engine.get_metadata("device")
    latent_dim = runtime_engine.get_metadata("latent_size")
    brain_shm = runtime_engine.get_shared_memory("brain")
    memory_manager_shm = runtime_engine.get_shared_memory("memory_manager")
    actuator_shm = runtime_engine.get_shared_memory("actuator")

    brain_cfg = cfg["brain"]
    module = brain_cfg["module"]
    ctx_len = brain_cfg["ctx_len"]
    # nu = runtime_engine.get_metadata("robot_props")["n_actuators"]
    nu = 3 # treat output as x, y, and orientation

    if module == "titans":
        brain = Titans(
            latent_dim,
            brain_cfg["titans"]["chunk_size"],
            device,
            nu,
        ).to(device)
    else:
        brain = LSTM(
            latent_dim,
            brain_cfg["lstm"]["hidden_dim"],
            brain_cfg["lstm"]["n_layers"],
            device,
            1,
            nu,
        ).to(device)

    brain_optimizer = torch.optim.Adam(brain.parameters(), lr=0.001)

    ctx = torch.zeros(
        (
            1,
            ctx_len,
            latent_dim,
        ),
        dtype=torch.float32,
    ).to(device)

    while True:
        latent = try_get(brain_shm["latent"], device)
        ctx = ctx.detach()
        if latent is None:
            continue
        ctx = fifo(ctx, latent)

        command_guidance = try_get(brain_shm["command"], device)
        emotion_guidance = try_get(brain_shm["emotion"], device)

        out = brain(ctx)
        logger.info(out)
        out_command = out["command"]
        out_emotions = out["emotions"]

        loss: int | torch.Tensor = 0

        if command_guidance is not None:
            command_loss = F.mse_loss(out_command, command_guidance)
            loss += command_loss

        if emotion_guidance is not None:
            emotions_loss = F.mse_loss(out_emotions, emotion_guidance)
            loss += emotions_loss

        if not (command_guidance is None) or not (emotion_guidance is None):
            brain_optimizer.zero_grad()
            if isinstance(loss, torch.Tensor):
                loss.backward()
            brain_optimizer.step()

        out_command.detach_()
        out_emotions.detach_()

        logger.info("Brain output command: %s", out_command)
        logger.info("Brain output emotions: %s", out_emotions)
        memory_manager_shm["command"].put(out_command)
        memory_manager_shm["emotion"].put(out_emotions)
        actuator_shm["command"].put(out_command)
