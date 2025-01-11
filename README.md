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

### Clone the repository
The easiest way to clone the repository is to create a ssh key and use it to clone the repository through ssh.
```bash
export SSHKEY=~/.ssh/id_ed25519_second
export GIT_SSH_COMMAND="ssh -i $SSHKEY"
git clone git@github.com:Simsreal/simsreal.git
cd simsreal
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

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Install Unity
*Optional* In progress of migration to Unity. Stay Tuned~!

### Environment Variables
#### Windows
Go to `Edit the system environment variables` and add `PYTHON_IS` and append Simsreal repository path to `PATH`

* `PYTHON_IS` = `C:\Users\<USERNAME>\AppData\Local\ov\pkg\isaac-sim-4.2.0\python.bat`

<!-- #### Linux -->

## Contribution
View [CONTRIBUTING.md](CONTRIBUTING.md) for more details on contribution to Simsreal.

## Launch Simsreal
### Memory
```bash
# windows
docker run --gpus all -p 6333:6333 `
    -v ${PWD}/qdrant_storage:/qdrant/storage `
    qdrant/qdrant:gpu-amd-latest

# linux
docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```
### Simulator
```bash
python simulators/simulators/aji6_simulator.py
```
**Notice: migrating to Unity**

### Simsreal
```bash
python host2.py
```

To understand the flow of consciousness emergence, you can have a look at [high-level flowchart](https://github.com/Simsreal/human/blob/main/src/images/flow_draft_2.png).
