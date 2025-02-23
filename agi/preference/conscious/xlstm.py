import torch
import torch.nn as nn
from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)

from agi.preference.efforts import Torques
from agi.preference.emotions import PAD


class xLSTM(nn.Module):
    def __init__(
        self,
        ctx_len,
        latent_size,
        n_blocks,
        device,
        n_actuators,
        slstm_at=[1],
        n_lstm_heads=4,
        convid_kernel_size=4,
        qkv_proj_blocksize=4,
        proj_factor=1.3,
    ):
        super().__init__()
        self.ctx_len = ctx_len
        self.latent_size = latent_size
        self.device = device

        xlstm_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=convid_kernel_size,
                    qkv_proj_blocksize=qkv_proj_blocksize,
                    num_heads=n_lstm_heads,
                ),
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=device,
                    num_heads=n_lstm_heads,
                    conv1d_kernel_size=convid_kernel_size,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=proj_factor,
                    act_fn="gelu",
                ),
            ),
            context_length=ctx_len,
            num_blocks=n_blocks,
            embedding_dim=latent_size,
            slstm_at=slstm_at,
        )
        self.xlstm_stack = xLSTMBlockStack(xlstm_cfg)
        self.torques = Torques(latent_size, n_actuators)
        self.pad = PAD(latent_size)

    def forward(self, ctx):
        out = self.xlstm_stack(ctx)
        out = out[:, -1, :]
        torques = self.torques(out)
        emotions = self.pad(out)
        assert not torch.any(torch.isnan(torques)) and not torch.any(
            torch.isnan(emotions)
        ), f"no nan is accepted: {ctx}"
        return {
            "torques": torques,
            "emotions": emotions,
        }
