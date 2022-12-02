import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, LeakyReLU, Reshape, Conv2DTranspose, BatchNormalization


class gan(keras.Model):
    def __init__(self):
        super(gan,self).__init__()
        self.a = 2
        self.gen = self.create_gen(32, None)
        self.dis = self.create_dis([28,28,1])
    
    def compile(self, d_opti, g_opti, run_eagerly = True):
        super(gan,self).compile(run_eagerly= run_eagerly)
        self.d_opti =  d_opti
        self.g_opti = g_opti
        self.loss = keras.losses.BinaryCrossentropy()

    def create_dis(self, input_size):
        dis = keras.Sequential()
        dis.add(keras.Input(shape=input_size))
        dis.add(Conv2D(32,4,4, 'same', activation=LeakyReLU(0.2)))
        dis.add(Conv2D(64,4,4, 'same', activation=LeakyReLU(0.2)))
        dis.add(Conv2D(128,4,2, 'same', activation=LeakyReLU(0.2)))
        dis.add(Dense(1,activation="sigmoid"))
        dis.add(Reshape((1,)))
        return dis

    def create_gen(self, input_size, output_size):
        gen = keras.Sequential()
        gen.add(keras.Input(shape=(input_size)))
        gen.add(Dense(14*14*128, activation= LeakyReLU(0.2)))
        gen.add(Reshape([14,14,128]))

        gen.add(Conv2DTranspose(128, 4, 1, 'same', activation=LeakyReLU(0.2)))
        gen.add(BatchNormalization())

        gen.add(Conv2DTranspose(64, 4, 2, 'same', activation=LeakyReLU(0.2)))
        gen.add(BatchNormalization())

        gen.add(Conv2DTranspose(32, 4, 1, 'same', activation=LeakyReLU(0.2)))
        gen.add(BatchNormalization())

        gen.add(Conv2DTranspose(1, 4, 1, 'same', activation=LeakyReLU(0.2)))
        gen.add(BatchNormalization())

        gen.add(Reshape([28,28]))

        gen.summary()

        return gen

    def train_step(self, data):
        x = data
        bs = x.shape[0]

        # training the discriminator
        self.gen.trainable = False
        self.dis.trainable = True

        noise = tf.random.normal(shape=[bs,32])
        generated = self.gen(noise)
        combined = tf.concat((generated,x), axis = 0)

        y = tf.concat([tf.ones((bs,1)), tf.zeros((bs,1))], axis = 0)

        with tf.GradientTape() as tape:
            y_pred = self.dis(combined)
            dis_loss = self.loss(y,y_pred)
        grads = tape.gradient(dis_loss, self.dis.trainable_weights)
        self.d_opti.apply_gradients(zip(grads, self.dis.trainable_weights))
        
        # training the generator
        self.gen.trainable = True
        self.dis.trainable = False

        noise = tf.random.normal(shape=[bs,32])
        y2 = tf.ones((bs,1))

        with tf.GradientTape() as tape:
            y_preds = self.dis(self.gen(noise))
            gen_loss = self.loss(y2, y_preds)

        grads = tape.gradient(gen_loss, self.gen.trainable_weights)
        self.d_opti.apply_gradients(zip(grads, self.gen.trainable_weights))


        return {"d_loss": dis_loss, "g_loss": gen_loss}


    def call(self,x ):
        return self.gen(x)
    