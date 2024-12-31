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
**Important** Please make sure you are running `aji5` or any humanoids following the orientation system of `aji5`.

### Compute Server
You must run compute server before next steps if you want to enable *consciousness*.
```bash
uvicorn compute_server.app:app --host 0.0.0.0 --port 8301 --workers 4
```

### Simulator
You must run simulator if you want to enable *consciousness*.
```bash
python simulators/simulators/aji5_simulator.py
```
**Notice: migrating to Unity**

### Simsreal
```bash
python host.py --config adji5 # simulation
python host.py -uc -s # silent, unconscious (no physics step)
```

To understand the flow of consciousness emergence, you can have a look at [high-level flowchart](https://github.com/Simsreal/human/blob/main/src/images/flow_draft_2.png).
