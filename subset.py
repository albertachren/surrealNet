import datasets
import numpy as np
from PIL import Image, ImageChops, ImageStat
import os

SOURCE_PATH = "sources/subset/"
BASE_PATH = "E:/Stuff/Python/datasets/surrealNet/"
SHOW = True
# Get range of files
# Get a file
# res/subsize
# remove extra
# get subset
# resize to 56x56
# add to array
# return array


def cut_cutoff(image, subsize):
    global SHOW
    cutoffx = image.size[0] % subsize
    cutoffy = image.size[1] % subsize
    area = (0, 0, image.size[0] - cutoffx, image.size[1] - cutoffy)
    image = image.crop(area)
    if SHOW is True:
        SHOW = False
    return image


def image_iterator(xrange, files, size, subsize):
    global SHOW
    c = 0
    if xrange[1] > len(files):
        xrange[1] = len(files)
    array = np.ndarray((0, size, size))
    for x in range(xrange[0], xrange[1]):
        try:
            source = Image.open(SOURCE_PATH + files[x]).convert('L')
            source = cut_cutoff(source, subsize)
            for xaxis in range(int(source.size[0] / subsize)):
                for yaxis in range(int(source.size[0] / subsize)):
                    c += 1
                    image = source.crop(
                        (xaxis * subsize, yaxis * subsize, xaxis *
                         subsize + subsize, yaxis * subsize + subsize))
                    image = image.resize((size, size), Image.ANTIALIAS)
                    backup = image
                    image = ImageChops.invert(image)
                    image = np.asarray(image)
                    image = np.expand_dims(image, 0)
                    if 50.0 < ImageStat.Stat(backup)._getmean()[0] < 200.0:
                        array = np.append(array, image, axis=0)
                    else:
                        pass
        except Exception as e:
            print("Bad something")
    print("x: ", x)
    print("c: ", c)
    return array


def generate(name, subsize, size):
    files = os.listdir(SOURCE_PATH)
    np.random.shuffle(files)
    samples = len(files)
    trainrange = round(samples / 2)
    array = image_iterator([0, samples], files, size, subsize)
    print(array.shape)
    xtrain = array[:round(len(array[:, 0, 0]) / 2), :, :]
    xtest = np.ones(len(array[:, 0, 0] / 2))
    ytrain = array[round(len(array[:, 0, 0]) / 2):(len(array[:, 0, 0])), :, :]
    ytest = np.ones(len(array[:, 0, 0]) - (len(array[:, 0, 0] / 2)))

    datasets.shuffle_array(xtrain, 420)
    datasets.shuffle_array(ytrain, 456)
    print(xtrain.shape)
    print(ytrain.shape)
    np.savez_compressed(os.path.join(BASE_PATH, 'subdataset' + "_x{}_{}px_i{}_ss{}_".format(xtrain.shape[0], size, samples, subsize) + name + '.npz'),
                        X_train=xtrain, y_train=ytrain, X_test=xtest, y_test=ytest)


if __name__ == '__main__':
    subsize = int(input("Subsize: "))
    size = int(input("Size: "))
    name = input("Name: ")
    generate(name, subsize, size)
