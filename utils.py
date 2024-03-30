import matplotlib.pyplot as plt
import numpy as np

# General
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
