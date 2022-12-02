import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

def normalize(im):
    return im/255.0

def denorm(im):
    return (im*255).astype(np.uint8)

def affiche_image(gann,n = 10):
    images = np.uint8(gann(n)*255)
    plt.figure(figsize=(15,10))
    for i, im in enumerate(images) :
        plt.subplot(1, n, i+1)
        plt.imshow(im, cmap = 'gray')
    plt.show()

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, GAN):
        self.gan = GAN

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            affiche_image(self.gan, 10)