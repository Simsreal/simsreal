[![Slack](https://img.shields.io/badge/slack-join%20chat-yellow.svg)](https://join.slack.com/t/simsreal/shared_invite/zt-2vwyklm9d-ppni~ex4pc4~t~5sBGpwFw)
[![Jira](https://img.shields.io/badge/jira-view%20project-blue.svg)](https://simsreal.atlassian.net/jira/software/c/projects/SR/boards/4?assignee=712020%3Acbb6a13b-ccf1-4d9d-8f59-7c4584c2d4ca)
## Table of Contents
- [Prerequisites](#prerequisites)
  - [Clone the repository](#clone-the-repository)
  - [Install development packages](#install-development-packages)
  - [Install NVIDIA Isaac Sim](#install-nvidia-isaac-sim)
  - [Install ROS2 Humble](#install-ros2-humble)
  - [Install Prometheus and Grafana (optional)](#install-prometheus-and-grafana-optional)
- [Start Simsreal](#start-simsreal)

## Prerequisites

### Clone the repository
The easiest way to clone the repository is to create a ssh key and use it to clone the repository through ssh.
```bash
export SSHKEY=~/.ssh/id_ed25519_second
export GIT_SSH_COMMAND="ssh -i $SSHKEY"
git clone git@github.com:Simsreal/simsreal.git
git submodule update --init --recursive
git pull --recurse-submodules
```

| Submodule | Description |
|-----------|-------------|
| simsreal | Main repo |
| environment | 3D physics-based simulation environment, ROS2 bridge builds. Robot/ world configurations |
| intelligence | The foundations for consciousness, abstract classes as references when developing intelligence |
| human | Primarily focused development of intelligence in Simsreal |
| neuro_symbol | High-level symbolic reasoning for humans to reason and create actions from granular behaviors |
| web | Streamlit web app |
| utilities | Utility functions |

### Install development packages
```bash
pip install -r requirements-dev.txt
chmod +x setup.sh
./setup.sh
```

### Install NVIDIA Isaac Sim
Please install Isaac Sim version `4.2.0` for development.
Follow [official documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) to install Omniverse and Isaac Sim.

### Install ROS2 Humble
Please install ROS2 Humble for development.
Follow [official documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_ros.html) to install ROS2 Humble.

### Install Prometheus and Grafana
(Optional) Prometheus and grafana are used to visualize intelligence's metrics and context. Refer to [docs/prometheus_grafana.md](docs/prometheus_grafana.md) for more details.

# Start Simsreal
The backend of Simsreal handles the consciousness emergence.
```bash
python host.py
```

For rapid prototyping, we use streamlit as the web framework.
```bash
streamlit run web/streamlit/main.py
```
