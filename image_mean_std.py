import numpy as np
from os import listdir
from os.path import join, isdir
from glob import glob
import os
from PIL import Image

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset

def cal_dir_stat(root, CHANNEL_NUM):
    im_pths = os.listdir(root)
    pixel_num = 0 # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)
    for path in im_pths:
        pil_img = Image.open(os.path.join(root, path))
        if pil_img.mode == 'I':
            # uint16 PNG stored as 32-bit int (HU + 1024, range 0-4095)
            # Normalize to [0,1] to match the loading in SegData
            im = np.array(pil_img, dtype=np.float32) / 4095.0
        else:
            im = np.array(pil_img, dtype=np.float32) / 255.0
        if im.ndim == 2:
            im = im[:, :, np.newaxis]  # add channel dim for grayscale
        pixel_num += im.shape[0] * im.shape[1]
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    mean = channel_sum / pixel_num
    std = np.sqrt(channel_sum_squared / pixel_num - np.square(mean))
    mean = np.around(mean, decimals=4)
    std = np.around(std, decimals=4)

    return list(mean), list(std)
