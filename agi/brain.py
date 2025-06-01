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
    robot_props = runtime_engine.get_metadata("robot_props")
    brain_shm = runtime_engine.get_shared_memory("brain")
    memory_manager_shm = runtime_engine.get_shared_memory("memory_manager")
    actuator_shm = runtime_engine.get_shared_memory("actuator")

    brain_cfg = cfg["brain"]
    module = brain_cfg["module"]
    ctx_len = brain_cfg["ctx_len"]

    # Handle different robot types
    if robot_props.get("data_type") == "simple_agent":
        # For simple agent: movement commands [movement_x, movement_y, orientation]
        nu = 3
        logger.info("Brain configured for simple agent - 3D movement commands")
    else:
        # For MuJoCo robots with actuators
        nu = robot_props.get("n_actuators", 3)  # Default to 3 if not found
        logger.info(f"Brain configured for robot - {nu} actuators")

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

        # For simple agent, this will be movement commands instead of torques
        movement_guidance = try_get(
            brain_shm["torque"], device
        )  # Still using "torque" queue for compatibility
        emotion_guidance = try_get(brain_shm["emotion"], device)

        out = brain(ctx)
        logger.info(out)

        # For simple agent, these are movement commands, not torques
        out_movements = out[
            "torques"
        ]  # Keep same key name for compatibility with brain models
        out_emotions = out["emotions"]

        loss: int | torch.Tensor = 0

        if movement_guidance is not None:
            if robot_props.get("data_type") == "simple_agent":
                # For simple agent, ensure we have the right shape [batch, 3] for [x, y, orientation]
                if movement_guidance.shape[-1] != 3:
                    if movement_guidance.shape[-1] == 2:
                        # Pad with zero orientation if only 2D movement
                        movement_guidance = torch.cat(
                            [
                                movement_guidance,
                                torch.zeros(
                                    movement_guidance.shape[0],
                                    1,
                                    device=movement_guidance.device,
                                ),
                            ],
                            dim=-1,
                        )
                    elif movement_guidance.shape[-1] == 1:
                        # Pad with zeros if only 1D
                        movement_guidance = torch.cat(
                            [
                                movement_guidance,
                                torch.zeros(
                                    movement_guidance.shape[0],
                                    2,
                                    device=movement_guidance.device,
                                ),
                            ],
                            dim=-1,
                        )

                # Ensure output has the right shape too
                if out_movements.shape[-1] != 3:
                    out_movements = out_movements[:, :3]  # Take first 3 dimensions

            movement_loss = F.mse_loss(out_movements, movement_guidance)
            loss += movement_loss

            if robot_props.get("data_type") == "simple_agent":
                logger.debug(
                    f"Movement loss: {movement_loss.item():.6f}, "
                    f"Target: [{movement_guidance[0, 0]:.3f}, {movement_guidance[0, 1]:.3f}, {movement_guidance[0, 2]:.3f}], "
                    f"Output: [{out_movements[0, 0]:.3f}, {out_movements[0, 1]:.3f}, {out_movements[0, 2]:.3f}]"
                )

        if emotion_guidance is not None:
            emotions_loss = F.mse_loss(out_emotions, emotion_guidance)
            loss += emotions_loss

        if not (movement_guidance is None) or not (emotion_guidance is None):
            brain_optimizer.zero_grad()
            if isinstance(loss, torch.Tensor):
                loss.backward()
            brain_optimizer.step()

        out_movements.detach_()
        out_emotions.detach_()

        # Send movement commands (still using "torque" queue names for compatibility)
        memory_manager_shm["torque"].put(out_movements)
        memory_manager_shm["emotion"].put(out_emotions)
        actuator_shm["torque"].put(out_movements)

        if robot_props.get("data_type") == "simple_agent":
            logger.debug(
                f"Brain output - Movement: [{out_movements[0, 0]:.3f}, {out_movements[0, 1]:.3f}, {out_movements[0, 2]:.3f}]"
            )
