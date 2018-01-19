import timeit
from PIL import Image, ImageOps
import numpy as np
import random
import math
from datasets import load_dataset
import os 

def f(x):
    x = 784 * x
    x = math.sqrt(x)
    return x

print(os.listdir())