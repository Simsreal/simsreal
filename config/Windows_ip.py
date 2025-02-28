import subprocess
import sys

from loguru import logger


def get_wsl_ip():
    try:
        cmd = 'powershell.exe Get-NetIPAddress -InterfaceAlias "*WSL*" -AddressFamily IPv4 | Select-Object -ExpandProperty IPAddress'
        result = subprocess.run(cmd, capture_output=True, text=True)

        ip = result.stdout.strip()

        if not ip:
            logger.error("ERROR: WSL IP address not found")
            sys.exit(1)
        return ip
    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    logger.info(get_wsl_ip())
