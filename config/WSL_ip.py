import subprocess
import sys


def get_wsl_ip():
    cmd = "ip addr show eth0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1"
    try:
        wsl_ip = subprocess.check_output(cmd, shell=True).decode().strip()
        if not wsl_ip:
            print("ERROR: WSL IP not found")
            sys.exit(1)
        return wsl_ip
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)


print(f"{get_wsl_ip()}")
