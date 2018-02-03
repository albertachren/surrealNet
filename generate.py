from keras import backend as K
from keras import initializers
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import h5py
from PIL import Image, ImageOps
import numpy as np


def network_setup(size):
    adam = Adam(lr=0.0002, beta_1=0.5)
    dim2 = size * size
    generator = Sequential()
    generator.add(Dense(1024, input_dim=10,
                        kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(2048))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(4168))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(dim2, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=adam)
    return generator


def load_image(filepath, size):
    image = Image.open(filepath).resize((size, size))
    image = np.array(image)
    return image


def load_network(weights_path):
    model = network_setup(56)
    model.load_weights(weights_path)
    return model


def show_image(array):
    im = model.predict(array)
    im = im.reshape(1, 56, 56)
    im = (im.astype(np.float32) + 127.5) * 127.5
    im = im.astype(int)
    im = im - 16128
    im = (im - 256) * -1
    image = Image.fromarray(im[0])
    image.show()


if __name__ == '__main__':
    model = load_network(
        "C:/Users/Albert/VSCode/surrealNet/results/surrealNetv2_e6000_x56_b128_56px_multiplemaimes/models/gan_generator_epoch_6000.h5")
    user_input = 'n'
    if user_input.lower() == 'i':
        path = input('Image: ')
        filepath = path + '.png'
        array = load_image(filepath, 56)
        array = np.array(array)
        array = array[:, :, 0]
        array = array.reshape(1, 3136)
        show_image(array)
    else:
        noise = np.random.normal(0, 1, size=[100, 10])
        generatedImages = model.predict(noise)
        generatedImages = generatedImages.reshape(
            100, 56, 56)
        plt.figure(figsize=(10, 10))
        for i in range(generatedImages.shape[0]):
            plt.subplot(10, 10, i + 1)
            plt.imshow(generatedImages[i],
                       interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig("generated/" + 'memes')
