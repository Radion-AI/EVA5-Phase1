import numpy as np

def image_size(sample):
    return np.transpose(sample.data[0], (2, 0, 1)).shape

def cal_mean(sample):
    return tuple(np.mean(sample.data, axis=(0, 1, 2)) / 255)

def cal_std(sample):
    return tuple(np.std(sample.data, axis=(0, 1, 2)) / 255)