import os

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras import initializers
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
from datasets import choose_dataset
# pylint: disable=invalid-name
os.environ["KERAS_BACKEND"] = "tensorflow"


K.set_image_dim_ordering('th')

# Deterministic output.
# Tired of seeing the same results every time? Remove the line below.
# np.random.seed(1000)

# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim = 10
dim2 = 784
img_width = 28
plot_fig_size = 20
# CUDA TOOLKIT, QDNN, Tensorflow-gpu
(X_train, y_train), (X_test, y_test) = choose_dataset()
print("loading data")
xrange = len(X_train[0])
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
print(X_train.shape)
X_train = X_train[: xrange, :, :]
X_train = X_train.reshape(len(X_train), dim2)

epochs = int(input("Epochs: "))
bsize = int(input("Batch size: "))
name = input("Name: ")
DIRECTORY  = "results/" + "surrealNet0" + "_e{}_b{}_{}".format(epochs, bsize, name) + '/'
# print(X_train.shape)
# print(X_train)
# print(X_test)

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

generator = Sequential()
generator.add(Dense(256, input_dim=randomDim,
                    kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(LeakyReLU(0.2))
generator.add(Dense(512))
generator.add(LeakyReLU(0.2))
generator.add(Dense(1024))
generator.add(LeakyReLU(0.2))
generator.add(Dense(dim2, activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)

discriminator = Sequential()
discriminator.add(Dense(1024, input_dim=dim2,
                        kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# Combined network
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

dLosses = []
gLosses = []


# Plot the loss from each batch
def plotLoss(epoch, bsize, img_width, name):
    print("plotting loss")
    plt.figure(figsize=(10, 8))
    plt.scatter(np.arange(len(dLosses[0])), dLosses[0],c='r',label='Discriminitive loss')
    plt.plot(gLosses[0], label='Generative loss')
    d = np.polyfit(np.arange(len(dLosses[0])), dLosses[0], 3)
    g = np.polyfit(np.arange(len(gLosses[0])), gLosses[0], 3)
    pd = np.poly1d(d)
    pg = np.poly1d(g)
    yd = pd(np.arange(len(dLosses[0])))
    yg = pg(np.arange(len(gLosses[0])))
    plt.plot(np.arange(len(dLosses[0])), yd, 'b--')
    plt.plot(np.arange(len(gLosses[0])), yg, 'r--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(
        DIRECTORY + 'gan_loss_epoch_{}_bsize_{}_{}px_{}.png'.format(epoch, bsize, img_width, name))


def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    # Create a wall of generated MNIST images
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, img_width, img_width)
    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(
        DIRECTORY + 'gan_generated_image_e{}_x{}_b{}_{}_{}.png'.format(epoch, xrange, bsize, img_width, name))


def saveModels(epoch, img_width, name):
    # Save the generator and discriminator networks (and weights) for later use
    generator.save(
        DIRECTORY +  'models/gan_generator_epoch_{}_size_{}_{}.h5'.format(epoch, img_width, name))
    discriminator.save(
        DIRECTORY + 'models/gan_discriminator_epoch{}_size_{}_{}.h5'.format(epoch, img_width, name))


def train(epochs=1, batchSize=128):
    batchCount = X_train.shape[0] / batchSize
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)
    os.makedirs(DIRECTORY, exist_ok=True)
    os.makedirs(DIRECTORY + "models/", exist_ok=True)

    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)

        # Get a random set of input noise and images
        noise = np.random.normal(0, 1, size=[batchSize, randomDim])
        imageBatch = X_train[np.random.randint(
            0, X_train.shape[0], size=batchSize)]
        # Generate fake MNIST images
        generatedImages = generator.predict(noise)
        # print np.shape(imageBatch), np.shape(generatedImages)
        X = np.concatenate([imageBatch, generatedImages])

        # Labels for generated and real data
        yDis = np.zeros(2 * batchSize)
        # One-sided label smoothing
        yDis[:batchSize] = 0.9

        # Train discriminator
        discriminator.trainable = True
        dloss = discriminator.train_on_batch(X, yDis)  # TODO: FUCKUP

        # Train generator
        noise = np.random.normal(0, 1, size=[batchSize, randomDim])
        yGen = np.ones(batchSize)
        discriminator.trainable = False
        gloss = gan.train_on_batch(noise, yGen)

        # Store loss of most recent batch from this epoch
        try:
            dLosses.append(dloss)
            gLosses.append(gloss)
        except:
            print("loss fail")
            pass
        if e == 1 or e == 10 or e % int((epochs / 5)) == 0:
            
            plotGeneratedImages(e, figsize=(plot_fig_size, plot_fig_size))
            print("Generating images")
            saveModels(e, img_width, name)

    # Plot losses from every epoch
    plotLoss(e, bsize, img_width, name)
    print("Finished")


if __name__ == '__main__':
    train(epochs, bsize)
