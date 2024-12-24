[![Slack](https://img.shields.io/badge/slack-join%20chat-yellow.svg)](https://join.slack.com/t/simsreal/shared_invite/zt-2vwyklm9d-ppni~ex4pc4~t~5sBGpwFw)
[![Jira](https://img.shields.io/badge/jira-view%20project-blue.svg)](https://simsreal.atlassian.net/jira/software/c/projects/SR/boards/4?assignee=712020%3Acbb6a13b-ccf1-4d9d-8f59-7c4584c2d4ca)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)

## Table of Contents
- [Prerequisites](#prerequisites)
  - [Clone the repository](#clone-the-repository)
  - [Install development packages](#install-development-packages)
  - [Install NVIDIA Isaac Sim](#install-nvidia-isaac-sim)
  - [Install ROS2 Humble](#install-ros2-humble)
  - [Install Prometheus and Grafana](#install-prometheus-and-grafana)
- [Contribution](#contribution)
- [Launch Simsreal](#launch-simsreal)

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
To understand more about the submodules, have a look at [docs/submodules_overview.md](docs/submodules_overview.md).

### Install development packages
```bash
pip install -r requirements-dev.txt
chmod +x setup.sh
./setup.sh
```

## Install Unity
*Optional*

### Install NVIDIA Isaac Sim
*Optional* Please install Isaac Sim version `4.2.0` for development.
Follow [official documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) to install Omniverse and Isaac Sim.

### Install ROS2 Humble
*Optional* Please install ROS2 Humble for development.
Follow [official documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_ros.html) to install ROS2 Humble.

### Install Prometheus and Grafana
*Optional* Prometheus and grafana are used to visualize intelligence's metrics and context. Refer to [docs/prometheus_grafana.md](docs/prometheus_grafana.md) for more details.

### Environment Variables
#### Windows
Go to `Edit the system environment variables` and add `PYTHON_IS` and append Simsreal repository path to `PATH`

* `PYTHON_IS` = `C:\Users\<USERNAME>\AppData\Local\ov\pkg\isaac-sim-4.2.0\python.bat`

<!-- #### Linux -->

## Contribution
View [CONTRIBUTING.md](CONTRIBUTING.md) for more details on contribution to Simsreal.

## Launch Simsreal
### Compute Server
Computer server must be started before next steps. It is very powerful too to provide multi-processing for compute-intensive tasks.
```bash
uvicorn compute_server.cpu.app:app --host 0.0.0.0 --port 8301 --workers 4
```

### Simulator
```bash
python simulators/launchers/mujoco/grace.py
```

### Simsreal
```bash
# isaac-sim
python host.py --config grace # or any other config under humanconfig/
```

To understand the flow of consciousness emergence, you can have a look at [high-level flowchart](https://github.com/Simsreal/human/blob/main/src/images/flow_draft_2.png).
