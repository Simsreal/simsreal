from queue import Empty

import torch


def try_get(queue, device) -> torch.Tensor | None:
    try:
        return queue.get_nowait().to(device)
    except Empty:
        return None
