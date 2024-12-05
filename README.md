# Simsreal
Consciousness emerges

## Prerequisites

### Clone the repository
The easiest way to clone the repository is to create a ssh key and use it to clone the repository through ssh.
```bash
# create a sshkey for the first time
export SSHKEY=~/.ssh/id_ed25519_second
export GIT_SSH_COMMAND="ssh -i $SSHKEY"
GIT_SSH_COMMAND="ssh -i $SSHKEY" git clone git@github.com:Simsreal/simsreal.git
GIT_SSH_COMMAND="ssh -i $SSHKEY" git submodule update --init --recursive
GIT_SSH_COMMAND="ssh -i $SSHKEY" git pull --recurse-submodules
```

### Install development packages
```bash
pip install -r requirements-dev.txt
chmod +x setup.sh
./setup.sh
```

### Install NVIDIA Isaac Sim
Please install Isaac Sim version `4.2.0` for development.
Follow [official documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) to install Omniverse and Isaac Sim.

### Install ROS2 Humble (Windows/ Linux)
Please install ROS2 Humble for development.
Follow [official documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_ros.html) to install ROS2 Humble.

### Install Prometheus and Grafana (optional)
Prometheus and grafana are used to visualize intelligence's metrics and context.
* TODO: integrate with ROS2 topics

To install prometheus, download from [here](https://prometheus.io/download/) and follow the instructions below.

```bash
tar xvf prometheus-*.tar.gz
cd prometheus-*
```
You can find sample `prometheus.yml` in the `prometheus-*` directory. There is a default one in [here](./testing/prometheus.yml).
start prometheus
```bash
./prometheus --config.file=prometheus.yml
```
start the app
```bash
python host.py
```
check `http://localhost:8000/metrics` to see if the metrics are being scraped.


visit `http://localhost:3000` to start create dashboards.

# Start Simsreal
The backend of Simsreal handles the consciousness emergence.
```bash
python host.py
```

For rapid prototyping, we use streamlit as the web framework.
```bash
streamlit run web/streamlit/main.py
```

# Interesting papers
* https://babyagi.org/
* https://github.com/yoheinakajima/babyagi
* https://research.a-star.edu.sg/articles/highlights/robot-olivias-lessons-in-tool-mastery/

# Recommended reading
* https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_omnigraph_shortcuts.html
