[![Slack](https://img.shields.io/badge/slack-join%20chat-yellow.svg)](https://join.slack.com/t/simsreal/shared_invite/zt-2vwyklm9d-ppni~ex4pc4~t~5sBGpwFw)
[![Jira](https://img.shields.io/badge/jira-view%20project-blue.svg)](https://simsreal.atlassian.net/jira/software/c/projects/SR/boards/4?assignee=712020%3Acbb6a13b-ccf1-4d9d-8f59-7c4584c2d4ca)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)

[![Papers](https://img.shields.io/badge/papers-view%20papers-red.svg?logo=data:image/svg+xml;base64,{{base64_encoded_svg}})](https://drive.google.com/drive/folders/1zibjXAV8tQxq0kdpxF_AjA39VclPqVNt)
[![Milestones](https://img.shields.io/badge/milestones-view%20milestones-yellow.svg?logo=data:image/svg+xml;base64,{{base64_encoded_svg}})](https://docs.google.com/spreadsheets/d/1sCcmwhLJEu_IFtE7pJBCcff9lTlVZqQWbb0RlZdycaw/edit?gid=0#gid=0)

## Table of Contents
- [Prerequisites](#prerequisites)
  - [Clone the repository](#clone-the-repository)
  - [Dependencies](#dependencies)
- [Contribution](#contribution)
- [Launch](#launch)

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
### Docker
```bash
docker build -t simsreal .
```
### Local

#### Install dependencies
```bash
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Contribution
View [CONTRIBUTING.md](CONTRIBUTING.md) for more details on contribution to Simsreal.

## Launch

### Docker
```bash
docker compose up
```

### Local

#### Simulator
Follow [Launch Unity](https://github.com/Simsreal/simulator?tab=readme-ov-file#launch-unity) to launch the simulator.

#### Simsreal
```bash
bash run_linux.sh # on linux
python run_wsl.py # on windows (WSL2)
```


## Performance

* ### CUDA MPS
Only available on Native Linux.

```bash
sudo su
# ====== launch =========
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
# ====== check =========
ps -ef | grep mps
# ====== stop =========
nvidia-smi -i 0 -c DEFAULT
echo quit | nvidia-cuda-mps-control
```
