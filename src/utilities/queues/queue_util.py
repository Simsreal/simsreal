from typing import Any

from queue import Empty


def try_get(queue, device=None) -> Any:
    try:
        if device is None:
            return queue.get_nowait()
        return queue.get_nowait().to(device)
    except Empty:
        return None
