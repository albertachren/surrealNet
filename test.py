"""Purely for testing"""
import timeit
from PIL import Image, ImageOps, ImageStat
import numpy as np
import random
import math
from datasets import choose_dataset
import os 


def f(x):
    x = 784 * x
    x = math.sqrt(x)
    return x

def shuffle_array(array, seed):
    np.random.seed(seed)
    np.random.shuffle(array)
    return array

image = Image.open("123.png")
a = ImageStat.Stat(image)._getmean()
print(a)