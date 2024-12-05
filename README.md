# Simsreal
Consciousness emerges

## Prerequisites
Install development packages (rapidly changing during early development)
```bash
pip install -r requirements-dev.txt
```

# Promethus and Grafana
## installation

prometheus - [Download page](https://prometheus.io/download/)
```bash
tar xvf prometheus-*.tar.gz
cd prometheus-*
```

Edit `prometheus.yml` inside. Sample:
```yaml
# my global config
global:
  scrape_interval: 15s # Set the scrape interval to every 15 seconds. Default is every 1 minute.
  evaluation_interval: 15s # Evaluate rules every 15 seconds. The default is every 1 minute.
  # scrape_timeout is set to the global default (10s).

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093

# Load rules once and periodically evaluate them according to the global 'evaluation_interval'.
rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

# A scrape configuration containing exactly one endpoint to scrape:
# Here it's Prometheus itself.
scrape_configs:
  # The job name is added as a label `job=<job_name>` to any timeseries scraped from this config.
  - job_name: "prometheus"

    # metrics_path defaults to '/metrics'
    # scheme defaults to 'http'.

    static_configs:
      - targets: ["localhost:9090"]

  - job_name: "human_app"
    static_configs:
      - targets: ["localhost:8000"]

```

start prometheus
```bash
./prometheus --config.file=prometheus.yml
```

start the app
```bash
python host.py
```

check `http://localhost:8000/metrics` to see if the metrics are being scraped.

grafana (linux)
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
