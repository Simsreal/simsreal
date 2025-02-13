import os
import shutil
import sys


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


def write_to_env(wsl_ip, windows_ip):
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
            print("ERROR: MCJF Path does not exist, please replace it")
            sys.exit(1)

        # Read existing .env file content
        env_content = ""
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                env_content = f.read()

        # Prepare new environment variables content
        lines = env_content.splitlines()
        new_lines = []
        wsl_ip_found = False
        windows_ip_found = False

        # Iterate through each line, replace or add WSL_IP and WINDOWS_IP
        for line in lines:
            if line.startswith("WSL_IP="):
                new_lines.append(f"WSL_IP={wsl_ip}")
                wsl_ip_found = True
            elif line.startswith("WINDOWS_IP="):
                new_lines.append(f"WINDOWS_IP={windows_ip}")
                windows_ip_found = True
            else:
                new_lines.append(line)

        # If variables not found, append them to the end of file
        if not wsl_ip_found:
            new_lines.append(f"WSL_IP={wsl_ip}")
        if not windows_ip_found:
            new_lines.append(f"WINDOWS_IP={windows_ip}")

        # Write updated content
        with open(env_path, "w") as f:
            f.write("\n".join(new_lines))

        print(f"WSL IP: {wsl_ip}")
        print(f"Windows IP: {windows_ip}")
    except Exception as e:
        print(f"ERROR: Failed to write .env file - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Get IP addresses from command line arguments or other methods
    wsl_ip = "172.27.160.28"  # Replace with actual WSL IP retrieval code
    windows_ip = "10.68.27.230"  # Replace with actual Windows IP retrieval code
    write_to_env(wsl_ip, windows_ip)
