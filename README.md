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
```


### Install dependencies
```bash
pip install -r requirements-dev.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# only on linux
pip install pylibraft-cu11 --extra-index-url=https://pypi.nvidia.com
pip install raft-dask-cu11 --extra-index-url=https://pypi.nvidia.com
```

## Contribution
View [CONTRIBUTING.md](CONTRIBUTING.md) for more details on contribution to Simsreal.

## Launch

### Simulator
Follow [Launch Unity](https://github.com/Simsreal/simulator?tab=readme-ov-file#launch-unity) to launch the simulator.

### Simsreal
```bash
python main.py
```
