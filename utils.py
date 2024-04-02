import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os
from datetime import datetime

# General
def setup_seed(seed=3407):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # 选择确定性算法
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.enabled = False

def log(loss_list, log_file):
    with open(log_file, 'w', encoding='utf-8') as file:
        content = ''
        for loss in loss_list:
            content += f"{loss} "
        content += f"\nRecorded on: {datetime.now()}\n"
        file.write(content)

def draw_figure(x, y, title, save_path):
    plt.plot(x, y)
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()

def convert_seconds(seconds):
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)

    hours = seconds // 3600
    seconds %= 3600

    minutes = seconds // 60
    seconds %= 60

    remaining_seconds = seconds
    return f"{days}d {hours}h {minutes}m {remaining_seconds}s"

# CV tasks RGB
def normalize_image(image):
    image = image.astype(np.float32)
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return (image / 255. - mean) / std

def unnormalize_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return (image * std + mean) * 255.
