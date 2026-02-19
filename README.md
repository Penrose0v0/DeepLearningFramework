# Deep Learning Framework

This is a commonly used deep learning framework (or maybe template? ).

My purpose in creating this repo is to enable better and faster initiation of deep learning tasks based on PyTorch. 

# Getting Started

Build docker image: 
```bash
docker build . -t dl_normal
```

Build container python environment: 
```bash
cd /root/workspace
# pip install torch first
pip install -e packages/core
pip install -e .
```

# Make a new project
- Customize dockerfile & change image name
- Modify devcontainer setting
- Build new model