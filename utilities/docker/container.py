import subprocess


def running_containers() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
    )
    return result.stdout.splitlines()


if __name__ == "__main__":
    print(running_containers())
