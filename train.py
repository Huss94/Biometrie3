import tensorflow as tf
from model import *
import os
import numpy as np 
import matplotlib.pyplot as plt


def noramlize(im):
    return im/255.0

def denorm(im):
    return (im*255).astype(np.uint8)

def main():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = noramlize(x_train)

    def affiche_image(gan,n = 10):
        ind = np.random.ranint(x_train.shape, n)
        gener = []
        plt.figure()
        for i in ind:
            plt.subplot(1, n, i+1)
            gener.append(denorm(gan(x_train[i])))
        plt.plot()




    class MyCallback(tf.keras.callbacks.Callback):
        def __init__(self, gan):
            self.gan = gan

        def on_epoch_end(self, epoch, logs=None):
            if epoch % 1 == 0:
                affiche_image(gan, 10)


    print(x_train.shape)
    print(x_test.shape)

    GAN = gan()
    GAN.compile( d_opti= keras.optimizers.Adam(learning_rate=0.0002), g_opti= keras.optimizers.Adam(learning_rate=0.0002))

    batch_size = 32

    history = GAN.fit(x = x_train, validation_split = 0.2, batch_size=batch_size, epochs = 5, callback = MyCallback(GAN))
    GAN.save_weights("models/GAN")
    return GAN

main()