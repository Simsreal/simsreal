import subprocess

from loguru import logger

def running_containers() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
    )
    return result.stdout.splitlines()


if __name__ == "__main__":
    logger.info(running_containers())
