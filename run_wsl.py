import os
import shutil
import sys
import subprocess
from dotenv import load_dotenv, set_key
from typing import Tuple


def get_mjcf_path(env_path):
    try:
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("MCJF_PATH="):
                    path = line.split("=")[1].strip().strip("'").strip('"')
                    # Check if the path exists in the system
                    if os.path.exists(path):
                        return path
                    return None
    except Exception as e:
        print(f"ERROR: Failed to find the path of MJCF in .env file - {str(e)}")


def get_ip_addresses() -> Tuple[str, str]:
    """
    Run the config/run.sh script and get WSL and Windows IP addresses
    Returns tuple of (wsl_ip, windows_ip)
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, "config", "run.sh")

        # Make sure the script is executable
        os.chmod(script_path, 0o755)

        # Run the script and get output
        result = subprocess.run(
            [script_path], capture_output=True, text=True, check=True
        )

        # Split the output into lines and get the IPs
        lines = result.stdout.strip().split("\n")
        if len(lines) != 2:
            raise ValueError("Expected 2 lines of output from run.sh")

        wsl_ip, windows_ip = lines
        return wsl_ip.strip(), windows_ip.strip()
    except Exception as e:
        print(f"Error getting IP addresses: {str(e)}")
        raise


def write_to_env():
    try:
        # Get the parent directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(current_dir, ".env")
        env_example_path = os.path.join(current_dir, ".env.example")

        # Copy .env.example to .env if .env doesn't exist but .env.example does
        if not os.path.exists(env_path) and os.path.exists(env_example_path):
            shutil.copy2(env_example_path, env_path)
            print(f"Copied .env.example to .env at: {env_path}")

        # Check if MCJF_PATH exists in the system
        mjcf_path = get_mjcf_path(env_path)
        if not mjcf_path:
            print("ERROR: MJCF Path does not exist, please replace it")
            sys.exit(1)

        # Load the .env file
        load_dotenv(env_path)

        # Get IP addresses from run.sh
        try:
            wsl_ip, windows_ip = get_ip_addresses()

            # Update the .env file with new IP addresses
            set_key(env_path, "WSL_IP", wsl_ip)
            set_key(env_path, "WINDOWS_IP", windows_ip)

            print(f"Updated .env file with WSL_IP={wsl_ip} and WINDOWS_IP={windows_ip}")

        except Exception as e:
            print(f"Warning: Failed to update IP addresses - {str(e)}")
            # Don't exit here, as this might be optional

    except Exception as e:
        print(f"ERROR: Failed to write .env file - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Get IP addresses from command line arguments or other methods
    write_to_env()
    # 请将更新后的env替换部分同路径下的./config.ymal文件
    # 具体而言， ymal文件格式如下
    """
robot:
  sub:
    protocol: tcp
    ip: 127.0.0.1
    port: 5556
  pub:
    protocol: tcp
    ip: 127.0.0.1
    port: 5557
  mjcf_path: /home/spoonbobo/simulator/Assets/MJCF/humanoid.xml
  pose: arm_stretch
    """
    # 你需要替换为
    """
robot:
  sub:
    protocol: tcp
    ip: 0.0.0.0
    port: 5556
  pub:
    protocol: tcp
    ip: ${WSL_IP}
    port: 5557
  mjcf_path: ${MJCF_PATH}
  pose: arm_stretch
    """
