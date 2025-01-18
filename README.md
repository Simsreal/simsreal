[![Slack](https://img.shields.io/badge/slack-join%20chat-yellow.svg)](https://join.slack.com/t/simsreal/shared_invite/zt-2vwyklm9d-ppni~ex4pc4~t~5sBGpwFw)
[![Jira](https://img.shields.io/badge/jira-view%20project-blue.svg)](https://simsreal.atlassian.net/jira/software/c/projects/SR/boards/4?assignee=712020%3Acbb6a13b-ccf1-4d9d-8f59-7c4584c2d4ca)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)

## Table of Contents
- [Prerequisites](#prerequisites)
  - [Clone the repository](#clone-the-repository)
  - [Install development packages](#install-development-packages)
  - [Install Unity](#install-unity)
- [Contribution](#contribution)
- [Launch Simsreal](#launch-simsreal)

## Prerequisites

### Requirements
* Python >= 3.10
* Docker

### Clone the repository
The easiest way to clone the repository is to create a ssh key and use it to clone the repository through ssh.
```bash
git clone git@github.com:Simsreal/simsreal.git
cd simsreal
git submodule update --init --recursive
```
To understand more about the submodules, have a look at [docs/submodules_overview.md](docs/submodules_overview.md).

### Install development packages
```bash
pip install -r requirements-dev.txt
chmod +x setup.sh
./setup.sh
```

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pylibraft-cu11 --extra-index-url=https://pypi.nvidia.com
pip install raft-dask-cu11 --extra-index-url=https://pypi.nvidia.com
```

### Install Unity
*Optional* In progress of migration to Unity. Stay Tuned~!

### Environment Variables (for Intellisense)
#### Windows

#### Linux
```bash
export PYTHONPATH=$PYTHONPATH:/home/spoonbobo/gitrepo/simsreal
```

## Contribution
View [CONTRIBUTING.md](CONTRIBUTING.md) for more details on contribution to Simsreal.

## Launch Simsreal
### CUDA MPS (Only on Linux)
Enable it to enhance multi-processing performance on GPU (Only Linux).
```bash
bash start_mps.sh
```

```shell
# upon MPS launch, you should see nvidia-cuda-mps-server in processes.
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.120                Driver Version: 550.120        CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090 ...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   64C    P0             55W /  115W |    1097MiB /  16376MiB |     40%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      2628      G   /usr/lib/xorg/Xorg                              4MiB |
|    0   N/A  N/A     10925      C   nvidia-cuda-mps-server                         28MiB |
|    0   N/A  N/A     67657    M+C   /usr/bin/python                               350MiB |
|    0   N/A  N/A     67661    M+C   /usr/bin/python                               368MiB |
|    0   N/A  N/A     67662    M+C   /usr/bin/python                               334MiB |
+-----------------------------------------------------------------------------------------+
```

### Memory
```bash
# windows
docker run --gpus all -p 6333:6333 `
    -v ${PWD}/qdrant_storage:/qdrant/storage `
    qdrant/qdrant:gpu-amd-latest

# linux
docker run --rm -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage  qdrant/qdrant
```
### Simulator
```bash
python simulators/simulators/aji6_simulator.py
```
**Notice: migrating to Unity**

### Simsreal
```bash
python main.py
```
