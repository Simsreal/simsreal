import os
import shutil
import sys
import subprocess
from typing import Tuple
import yaml

from loguru import logger
from dotenv import load_dotenv, set_key


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
        logger.error(f"ERROR: Failed to find the path of MJCF in .env file - {str(e)}")


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
        logger.error(f"Error getting IP addresses: {str(e)}")
        raise


def update_yaml_config(wsl_ip, mjcf_path):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "config.yaml")

        # 读取现有的YAML文件
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # 只更新robot部分的配置
        if "robot" in config:
            config["robot"].update(
                {
                    "sub": {"protocol": "tcp", "ip": "0.0.0.0", "port": 5556},
                    "pub": {"protocol": "tcp", "ip": wsl_ip, "port": 5557},
                    "mjcf_path": mjcf_path,
                    "pose": "arm_stretch",
                }
            )

        # 写回YAML文件，保持其他配置不变
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Updated robot configuration in config.yaml at: {yaml_path}")

    except Exception as e:
        logger.error(f"ERROR: Failed to update YAML config - {str(e)}")
        sys.exit(1)


def write_to_env():
    try:
        # Get the parent directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(current_dir, ".env")
        env_example_path = os.path.join(current_dir, ".env.example")
        yaml_path = os.path.join(current_dir, "config.yaml")
        yaml_example_path = os.path.join(current_dir, "config.template.yaml")

        # Copy .env.example to .env if .env doesn't exist but .env.example does
        if not os.path.exists(env_path) and os.path.exists(env_example_path):
            shutil.copy2(env_example_path, env_path)
            logger.info(f"Copied .env.example to .env at: {env_path}")

        # Copy config.yaml.example to config.yaml if config.yaml doesn't exist but config.yaml.example does
        if not os.path.exists(yaml_path) and os.path.exists(yaml_example_path):
            shutil.copy2(yaml_example_path, yaml_path)
            logger.info(f"Copied config.template.yaml to config.yaml at: {yaml_path}")

        # Check if MCJF_PATH exists in the system
        mjcf_path = get_mjcf_path(env_path)
        if not mjcf_path:
            logger.error("ERROR: MJCF Path does not exist, please replace it")
            sys.exit(1)

        # Get IP addresses from run.sh
        try:
            wsl_ip, windows_ip = get_ip_addresses()

            # Update the .env file with new IP addresses
            set_key(env_path, "WSL_IP", wsl_ip)
            set_key(env_path, "WINDOWS_IP", windows_ip)

            logger.info(
                f"Updated .env file with WSL_IP={wsl_ip} and WINDOWS_IP={windows_ip}"
            )

            # Update the YAML config file
            update_yaml_config(wsl_ip, mjcf_path)

        except Exception as e:
            logger.warning(f"Warning: Failed to update IP addresses - {str(e)}")
            # Don't exit here, as this might be optional

        # Load the .env file
        load_dotenv(env_path)
        mjcf_path = os.getenv(
            "MCJF_PATH", "/home/spoonbobo/simulator/Assets/MJCF/humanoid.xml"
        )

    except Exception as e:
        logger.error(f"ERROR: Failed to write .env file - {str(e)}")
        sys.exit(1)


def run_main():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        main_path = os.path.join(current_dir, "main.py")

        if not os.path.exists(main_path):
            logger.error(f"ERROR: no main.py found in {current_dir}")  # type: ignore
            sys.exit(1)

        # run main.py on python
        subprocess.run([sys.executable, main_path], check=True)

    except subprocess.CalledProcessError as e:
        logger.error(f"ERROR: Failed to run main.py - {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"ERROR: Unexpected error while running main.py - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Get IP addresses from command line arguments or other methods
    write_to_env()
    run_main()
