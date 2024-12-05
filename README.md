# Simsreal
Consciousness emerges

## Prerequisites

### Install development packages (rapidly changing during early development)
```bash
pip install -r requirements-dev.txt
```

### Install Prometheus and Grafana (optional)
Prometheus and grafana are used to visualize intelligence's metrics and context.
* TODO: integrate with ROS2 topics\

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

To install grafana (linux/ wsl2)
```bash
sudo apt-get install -y adduser libfontconfig1
wget https://dl.grafana.com/oss/release/grafana_8.5.2_amd64.deb
sudo dpkg -i grafana_8.5.2_amd64.deb

sudo systemctl daemon-reload
sudo systemctl start grafana-server
sudo systemctl enable grafana-server.service
```

visit `http://localhost:3000` and all set.

# Streamlit
```bash
streamlit run web/streamlit/main.py
```

# PDDL & Downward
```bash
sudo apt update
sudo apt install build-essential cmake git

git clone https://github.com/aibasel/downward.git
cd downward

./build.py
```

```bash
export SSHKEY=~/.ssh/id_ed25519_second
export GIT_SSH_COMMAND="ssh -i $SSHKEY"
GIT_SSH_COMMAND="ssh -i $SSHKEY" git clone git@github.com:Simsreal/simsreal.git
GIT_SSH_COMMAND="ssh -i $SSHKEY" git submodule update --init --recursive
GIT_SSH_COMMAND="ssh -i $SSHKEY" git pull --recurse-submodules
```

# ROS/ ROS2
Tutorial for ROS2 with Isaac Sim.
Must complete to get familiarize with ROS2 with Isaac Sim.

```bash
git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces

cd IsaacSim-ros_workspaces/humble_ws

colcon build
```

## Turtlebot3
```bash
git clone -b humble-devel https://github.com/ROBOTIS-GIT/turtlebot3.git turtlebot3
```


https://docs.omniverse.nvidia.com/isaacsim/latest/ros2_tutorials/index.html

# Interesting papers
* https://babyagi.org/
* https://github.com/yoheinakajima/babyagi
* https://research.a-star.edu.sg/articles/highlights/robot-olivias-lessons-in-tool-mastery/

# Recommended reading
* https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_omnigraph_shortcuts.html
