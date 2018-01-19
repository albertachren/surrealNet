from keras import backend as K
from keras import initializers
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
import h5py
from PIL import Image, ImageOps
import numpy as np


def network_setup(size):
    adam = Adam(lr=0.0002, beta_1=0.5)
    dim2 = size * size
    generator = Sequential()
    generator.add(Dense(256, input_dim=10,
                        kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(dim2, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=adam)
    return generator


def load_network(weights_path):
    model = network_setup(28)
    model.load_weights(weights_path)
    return model


if __name__ == '__main__':
    noise = np.random.normal(0, 1, size=[1, 10])
    model = load_network(
        "C:/Users/Albert/VSCode/surrealNet/results/surrealNet_e20000_b20_semisupertest2/models/gan_generator_epoch_20000_size_28_semisupertest2.h5")
    im = model.predict(noise)
    im = im.reshape(1, 28, 28)
    im = (im.astype(np.float32) + 127.5) * 127.5
    im = im.astype(int)
    im = im - 16128
    im = (im - 256) *-1
    image = Image.fromarray(im[0])
    image.show()
