"""NPZ Dataset management and creation"""
# Ask for props
# Generate images, half train and half test
# Convert to (compressed) npz file
# TODO: Ask about false data
import random
import numpy as np
from PIL import Image, ImageOps
import os

batch_size = 300
BASE_PATH = "E:/Stuff/Python/datasets/surrealNet/"
RESIZE_LIST = (1, 1, 1, 0.8, 0.8, 0.5)
show = True
# "E:\Stuff\Python\datasets\surrealNet"


def generate_image(source, offset, rotation, size, bcolor):
    """Return a single image in numpy format"""
    """source is an image already opened"""
    x = random.randrange(-offset, offset)
    y = random.randrange(-offset, offset)
    rs = int(random.choice(RESIZE_LIST) * size)
    global show
    #source = Image.open(source).resize((size, size), Image.ANTIALIAS)
    source = source.rotate(
        random.randrange(-rotation, rotation), resample=Image.BILINEAR)
    #source = source.resize((rs, rs), Image.ANTIALIAS)
    image = Image.new('RGB', (size, size), bcolor)
    image.paste(source, (x, y), source)
    image = image.convert('L')
    image = ImageOps.invert(image)
    if show:
        image.show()
        show = False
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, 0)
    return image_array


def generator(name, source, amount, offset, rotation, size, bcolor, false=False, false_split=0.33):
    """Generate and save a npz from properties"""
    if false:
        false_amount = amount * false_split
        xrange = amount - false_amount
    else:
        xrange = amount
    sourcef = Image.open(source).resize((size, size), Image.ANTIALIAS)
    xtrain = image_iterator(sourcef, int(xrange / 2),
                            offset, rotation, size, bcolor)
    xtest = image_iterator(sourcef, int(xrange / 2),
                           offset, rotation, size, bcolor)
    ytrain = np.ones(xrange)
    ytest = np.ones(xrange)
    np.savez_compressed(os.path.join(BASE_PATH, 'dataset' + "_x{}_{}px_".format(xrange, size) + name + '.npz'),
                        X_train=xtrain, y_train=ytrain, X_test=xtest, y_test=ytest)


def generator_multiple(name, amount, size, bcolor, rotation, false=False, false_split=0.33):
    """Generate and save a npz from properties"""
    if false:
        false_amount = amount * false_split
        xrange = amount - false_amount
    else:
        xrange = amount

    sources = []
    labels = []
    source_amount = int(input("Source amount: "))
    for x in range(source_amount):
        sources.append("sources/" + input("Source " + str(x+1) +  ": ").format((x)) + '.png')
        labels.append(input("Label: "))

    user_input = input("Switch X & Y?")
    print("Generating file...")
    
    ytrain = ()
    ytest = ()
    xtrain = np.ndarray((0,size,size))
    xtest = np.ndarray((0,size,size))
    for x in range(len(sources)):
        sourcef = Image.open(sources[x]).resize((size, size), Image.ANTIALIAS)
        xtrain = np.append(xtrain, image_iterator(sourcef, int(xrange / len(sources)),
                                                  round(size / 4), rotation, size, bcolor), axis=0)
        xtest = np.append(xtest, image_iterator(sourcef, int(xrange / len(sources)),
                                                round(size / 4), rotation, size, bcolor), axis=0)
        for _ in range(int(xrange / len(sources))):
            ytrain = np.append(ytrain, labels[x])
            ytest = np.append(ytest, labels[x])

    if user_input == 'y':
        temp = xtrain
        temp2 = ytrain
        ytrain = temp
        xtrain = temp2

        temp = xtest
        temp2 = ytest
        ytest = temp
        xtest = temp2
    else:
        pass

    np.savez_compressed(os.path.join(BASE_PATH, 'dataset' + "_x{}_{}px_".format(xrange, size) + name + '.npz'),
                        X_train=xtrain, y_train=ytrain, X_test=xtest, y_test=ytest)


def image_iterator(source, amount, offset, rotation, size, bcolor):
    c = 0
    array = np.ndarray((0, size, size))
    temp = np.ndarray((0, size, size))
    for x in range(0, amount):
        temp = np.append(temp, generate_image(
            source, offset, rotation, size, bcolor), axis=0)
        if c % batch_size == 0:
            array = np.append(array, temp, axis=0)
            temp = np.empty((0, size, size))
            print(c)
        c += 1
    return array


def load_dataset(filename):
    """Return the dataset from a npz file"""
    npz = np.load(os.path.join(BASE_PATH, filename))
    X_train = npz['X_train']
    y_train = npz['y_train']
    X_test = npz['X_test']
    y_test = npz['y_test']
    return(X_train, y_train), (X_test, y_test)


def generator_setup():
    NAME = input("Name: ")
    SOURCE = "sources/" + input("Source: ") + '.png'
    if SOURCE == 's':
        SOURCE = "source.png"
    AMOUNT = int(input("Amount total: "))
    SIZE = int(input("Size: "))
    OFFSET = int(input("Offset: "))
    ROTATION = int(input("Rotation: "))
    FALSE = input("False source: ")
    FALSE_SPLIT = 0.33  # Default
    if FALSE.lower() == "n":
        FALSE = False
    else:
        FALSE_SPLIT = int(input("False split %: ")) / 100

    BACKGROUND_COLOR = input("Background color (s/w/#): ")
    if BACKGROUND_COLOR == 's':
        BACKGROUND_COLOR = "#0e3966"
    if BACKGROUND_COLOR.lower() == 'w':
        BACKGROUND_COLOR = '#ffffff'

    user_input = input("Multiple surces? ")
    if user_input.lower() == 'y':
        generator_multiple(NAME, AMOUNT, SIZE, BACKGROUND_COLOR, ROTATION)
    else:
        print("Generating file...")
        generator(NAME, SOURCE, AMOUNT, OFFSET, ROTATION, SIZE,
              BACKGROUND_COLOR, FALSE, FALSE_SPLIT)


def choose_dataset():
    files = os.listdir(BASE_PATH)
    for x in range(len(files)):
        print(str(x) + '. ' + files[x])
    user_input = int(input("Choose dataset: "))
    print("loading data")
    return load_dataset(files[user_input])


if __name__ == '__main__':
    generator_setup()
