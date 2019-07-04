import numpy as np
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import AlphaDropout
from keras import initializers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

np.random.seed(70)

def build_sequential_model(input_rate, rate, shape):
    model = Sequential()

    model.add(AlphaDropout(input_rate, input_shape=(shape,)))

    model.add(Dense(6, activation="linear", kernel_initializer="lecun_normal"))
    model.add(Activation('selu'))
    model.add(AlphaDropout(rate))

    model.add(Dense(3, activation="linear", kernel_initializer="lecun_normal"))
    model.add(Activation('selu'))
    model.add(AlphaDropout(rate))

    model.add(Dense(units=1, activation="linear", kernel_initializer="lecun_normal"))

    optim = Adam(lr=0.01, beta_1=0.95)

    model.compile(loss='mean_squared_error',
                    optimizer=optim)
    return model


def fit_model_batch(model, x, y, num_epoch=None):
    if num_epoch is None:
        num_epoc = 1000
    es = [EarlyStopping(monitor='loss', min_delta=0, patience=200, verbose=0, mode='auto')]
    model.fit(x, y, epochs=num_epoch, batch_size=x.shape[0], verbose = 0, callbacks = es) #full batch size
    return model
