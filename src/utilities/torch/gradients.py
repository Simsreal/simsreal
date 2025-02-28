import torch.nn as nn

from loguru import logger

def check_gradients(model: nn.Module):
    """
    Utility function to check and print the gradient norms of a PyTorch model's parameters.

    Args:
        model (nn.Module): The PyTorch model to check.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            logger.info(f"Gradient for {name}: {param.grad.norm()}")
        else:
            logger.info(f"Gradient for {name} is None")
