import tensorflow as tf
from model import *
from utils import *
import numpy as np 





def main():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    all_data= np.concatenate((x_train, x_test), axis = 0)
    all_data= normalize(all_data)

    # Le GAN est un keras model redéfinie, (regardre model.py)
    GAN = gan()
    GAN.compile( d_opti= keras.optimizers.Adam(learning_rate=0.0002), g_opti= keras.optimizers.Adam(learning_rate=0.0002))

    batch_size = 64

    # Voir utils.py
    S = MyCallback(GAN)

    history = GAN.fit(x= all_data, validation_split = 0.2,  batch_size=batch_size, epochs = 20, callbacks = [S])
    GAN.save_weights("models/GAN")
    return GAN

main()