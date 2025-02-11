import torch.nn as nn


def check_gradients(model: nn.Module):
    """
    Utility function to check and print the gradient norms of a PyTorch model's parameters.

    Args:
        model (nn.Module): The PyTorch model to check.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name}: {param.grad.norm()}")
        else:
            print(f"Gradient for {name} is None")
