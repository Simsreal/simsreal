from typing import List

import pynvml
import psutil
from loguru import logger


def get_nvidia_process_names() -> List[str]:
    # Initialize NVML
    pynvml.nvmlInit()

    # Get the number of GPUs
    device_count = pynvml.nvmlDeviceGetCount()

    process_names = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        # Get the running processes on the GPU
        try:
            process_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for process in process_info:
                pid = process.pid
                try:
                    # Use psutil to get the process name
                    proc = psutil.Process(pid)
                    process_name = proc.name()
                    process_names.append(process_name)
                except psutil.NoSuchProcess:
                    logger.warning(f"Process with PID {pid} no longer exists.")
        except pynvml.NVMLError as error:
            logger.warning(f"Error getting processes for GPU {i}: {error}")

    # Shutdown NVML
    pynvml.nvmlShutdown()

    return process_names


# Example usage
if __name__ == "__main__":
    process_names = get_nvidia_process_names()
    logger.info("Running processes using NVIDIA GPU:")
    for name in process_names:
        logger.info(name)
