import torch
import torch.nn as nn
from torchvision import transforms

from agi.learning.perceive import Retina
from src.utilities.queues.queue_util import try_get

# from utilities.torch.gradients import check_gradients

vision_mean = [0.485, 0.456, 0.406]
vision_std = [0.229, 0.224, 0.225]


def vision_preproc(x) -> torch.Tensor | None:
    if x is None:
        return None
    return transforms.Compose(
        [
            transforms.Resize(128),
            transforms.CenterCrop((128, 128)),
        ]
    )(x)


def vae_loss_function(reconstructed, original, mu, logvar) -> torch.Tensor:
    reconstruction_loss = nn.functional.mse_loss(
        reconstructed, original, reduction="sum"
    )
    kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = reconstruction_loss + kl_divergence_loss
    return total_loss


def perceiver(runtime_engine, name):
    def forward(x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r, mu, logvar = perceptor(x)
        mu_normalized = mu / torch.linalg.norm(mu, ord=2, dim=1, keepdim=True)
        return r, mu_normalized, logvar

    def backprop(r, x0, mu, logvar) -> None:
        loss = vae_loss_function(r, x0, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    perceivers_lookup = {
        "vision": Retina,
    }
    preproc_lookup = {
        "vision": vision_preproc,
    }
    cfg = runtime_engine.get_metadata("config")
    perceivers_cfg = cfg["perceivers"]
    device = runtime_engine.get_metadata("device")
    perceptor = perceivers_lookup[name](emb_dim=perceivers_cfg[name]["emb_dim"]).to(
        device
    )

    optimizer = torch.optim.Adam(perceptor.parameters(), lr=0.001)

    stream = torch.cuda.Stream(device=device)
    perceiver_shm = runtime_engine.get_shared_memory("perceiver")
    memory_manager_shm = runtime_engine.get_shared_memory("memory_manager")

    while True:
        with torch.cuda.stream(stream):  # type: ignore
            preproc = preproc_lookup[name]
            ctx = try_get(perceiver_shm[name], device)

            if ctx is not None:
                x = preproc(ctx)
                if x is None:
                    continue
                x = x.to(device).unsqueeze(0)
                x0 = x.clone()
                r, mu_normalized, logvar = forward(x)
                if torch.any(torch.isnan(mu_normalized)):
                    continue
                backprop(r, x0, mu_normalized, logvar)
                memory_manager_shm[f"{name}_latent"].put(mu_normalized.detach())
