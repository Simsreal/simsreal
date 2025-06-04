from typing import Any

from queue import Empty


def try_get(queue, device=None, expect_tensor=True):
    try:
        item = queue.get_nowait()
        if expect_tensor and hasattr(item, "to"):
            return item.to(device)
        return item
    except:
        return None
