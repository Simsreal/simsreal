import time
from queue import Empty

import torch
import torch.nn.functional as F

from human.preference.conscious import LSTM, xLSTM


def brain_proc(runtime_engine):
    cfg = runtime_engine.get_metadata("config")
    device = runtime_engine.get_metadata("device")
    latent_dim = runtime_engine.get_shm("latent").shape[-1]

    def fifo(ctx, x) -> torch.Tensor:
        x = x.to(device)
        return torch.cat((ctx[:, 1:, :], x.unsqueeze(1)), dim=1)

    def try_get(queue) -> torch.Tensor | None:
        try:
            return queue.get_nowait().to(device)
        except Empty:
            return None

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
    stream = torch.cuda.Stream(device=device)

    ctx = torch.zeros(
        (
            1,
            ctx_len,
            latent_dim,
        ),
        dtype=torch.float32,
    ).to(device)

    while True:
        with torch.cuda.stream(stream):  # type: ignore
            ctx = ctx.detach()
            ctx = fifo(ctx, runtime_engine.get_shm("latent"))
            out = lstm(ctx)
            out_torques = out["torques"]
            out_emotions = out["emotions"]

            t = try_get(runtime_engine.get_queue("drives_q"))
            e = try_get(runtime_engine.get_queue("emotions_q"))

            loss = 0

            if t is not None:
                torque_loss = F.mse_loss(out_torques, t)
                loss += torque_loss

            if e is not None:
                emotions_loss = F.mse_loss(out_emotions, e)
                loss += emotions_loss

            if not (t is None) or not (e is None):
                brain_optimizer.zero_grad()
                loss.backward()  # type: ignore
                brain_optimizer.step()

            with torch.no_grad():
                runtime_engine.update_shm("torques", out_torques.detach())
                runtime_engine.update_shm("emotions", out_emotions.detach())

        time.sleep(1 / cfg["running_frequency"])
