// ... existing code ...

## Launch

### Docker
```bash
docker compose up
```

### Local

#### Simulator
Follow [Launch Unity](https://github.com/Simsreal/simulator?tab=readme-ov-file#launch-unity) to launch the simulator.

#### Simsreal

**Basic Launch:**
```bash
# Linux
bash run_linux.sh

# Windows (WSL2)
python run_wsl.py

# Direct execution
python main.py
```

**Advanced Options:**
```bash
# Enable debug frame saving
python main.py --debug-frames

# Use custom config file
python main.py --config custom_config.yaml

# Disable image generation for better performance
python main.py --disable-images

# Show exploration summary for a specific run
python main.py --exploration-summary 5

# List all available runs
python main.py --list-runs
```

// ... existing code ...