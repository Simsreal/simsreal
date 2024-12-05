# To install grafana (linux/ wsl2)
sudo apt-get install -y adduser libfontconfig1
wget https://dl.grafana.com/oss/release/grafana_8.5.2_amd64.deb
sudo dpkg -i grafana_8.5.2_amd64.deb

sudo systemctl daemon-reload
sudo systemctl start grafana-server
sudo systemctl enable grafana-server.service

sudo apt update
sudo apt install build-essential cmake git

git clone https://github.com/aibasel/downward.git

git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces
./downward/build.py
cd IsaacSim-ros_workspaces/humble_ws
# colcon build
