import numpy as np
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2
import timeit
import os

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset

def cal_dir_stat(root, CHANNEL_NUM):
    im_pths = os.listdir(root)
    pixel_num = 0 # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)
    for path in im_pths:
        im = cv2.imread(os.path.join(root,path)) # image in M*N*CHANNEL_NUM shape, channel in BGR order
        #print(root+path)
        im = im/255.0
        pixel_num += (im.size/CHANNEL_NUM)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))
 
    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
    bgr_mean = np.around(bgr_mean, decimals=4)
    bgr_std = np.around(bgr_std, decimals=4)
    
    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]
    
    return rgb_mean, rgb_std
