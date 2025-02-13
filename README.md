# Deep Learning Framework

This is a commonly used deep learning framework (or maybe template? ).

My purpose in creating this repo is to enable better and faster initiation of deep learning tasks based on PyTorch. 

# Getting Started

If you use docker, run: 
```
$ docker pull penrose0v0/dl_normal:latest
$ docker run --gpus all -it --net=host --shm-size=16gb --name -v /{data_dir}:/root/share --name {container_name} penrose0v0/dl_normal:latest
```

Next, build your python environment: 
```
% python -m venv {venv_path}
% source {venv_path}/bin/activate
% pip install -r requirements.txt
```

Then, modify the details of the code. 

Finally, start training by running `python train.py`.  

Also, some commonly used model architectures (such as U-Net, ResNet) are already prepared in `components.py`, and is still being updated. 