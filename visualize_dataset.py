import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
import math

import datasets


def plotImages(array, examples=100, dim=(10, 10), figsize=(10, 10)):
    plt.figure(figsize=figsize)
    try:
        for i in range(100):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(array[i, :, :], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
    except IndexError:
        pass
    plt.tight_layout()
    plt.show()

def rmsdiff(im1, im2): # Thanks StackOverflow
    "Calculate the root-mean-square difference between two images"

    h = ImageChops.difference(im1, im2).histogram()

    # calculate rms
    #return math.sqrt(reduce(operator.add, map(lambda h, i: h*(i**2), h, range(256))) / (float(im1.size[0]) * im1.size[1]))

def find_similars(array):
    for x in range(array.shape[0]):
        pass

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = datasets.choose_dataset()
    plotImages(X_train)
    plotImages(y_train)
