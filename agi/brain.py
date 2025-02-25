# import time

import torch
import torch.nn.functional as F

from agi.preference.conscious import LSTM, xLSTM
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
    nu = runtime_engine.get_metadata("robot_props")["n_actuators"]

    if brain_cfg["module"] == "xlstm" and device == "cuda":
        lstm = xLSTM(
            ctx_len=brain_cfg["lstm"]["ctx_len"],
            latent_size=latent_dim,
            n_blocks=brain_cfg["lstm"]["n_xlstm_blocks"],
            device=device,
            n_actuators=nu,
            n_lstm_heads=brain_cfg["lstm"]["n_lstm_heads"],
            convid_kernel_size=brain_cfg["lstm"]["convid_kernel_size"],
            qkv_proj_blocksize=brain_cfg["lstm"]["qkv_proj_blocksize"],
            proj_factor=brain_cfg["lstm"]["proj_factor"],
        ).to(device)

    else:
        lstm = LSTM(
            latent_dim,
            brain_cfg["lstm"]["hidden_dim"],
            brain_cfg["lstm"]["n_layers"],
            device,
            1,
            nu,
        ).to(device)

    brain_optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
    ctx_len = brain_cfg["lstm"]["ctx_len"]

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
        print(ctx.shape)

        torque_guidance = try_get(brain_shm["torque"], device)
        emotion_guidance = try_get(brain_shm["emotion"], device)

        out = lstm(ctx)
        out_torques = out["torques"]
        out_emotions = out["emotions"]

        loss: int | torch.Tensor = 0

        if torque_guidance is not None:
            torque_loss = F.mse_loss(out_torques, torque_guidance)
            loss += torque_loss

        if emotion_guidance is not None:
            emotions_loss = F.mse_loss(out_emotions, emotion_guidance)
            loss += emotions_loss

        if not (torque_guidance is None) or not (emotion_guidance is None):
            brain_optimizer.zero_grad()
            if isinstance(loss, torch.Tensor):
                loss.backward()
            brain_optimizer.step()

        out_torques.detach_()
        out_emotions.detach_()

        memory_manager_shm["torque"].put(out_torques)
        memory_manager_shm["emotion"].put(out_emotions)
        actuator_shm["torque"].put(out_torques)
