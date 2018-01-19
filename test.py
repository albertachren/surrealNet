"""Purely for testing"""
import timeit
from PIL import Image, ImageOps
import numpy as np
import random
import math
from datasets import choose_dataset
import os 


def f(x):
    x = 784 * x
    x = math.sqrt(x)
    return x

for x in range(30):
    print(f(x))
RESIZE_LIST = (1, 1, 1, 0.8, 0.8, 0.5)
print(random.choice(RESIZE_LIST)*2)

(a,b),(c,d) = choose_dataset()
print(b[0])
print(b[1])
print(b[2])
print("lol")
# 28, 56, 84, 112, 140