import numpy as np

from keras.models import Sequential
import numpy as np
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation

def get_data(file):
    return [line for line in open(file)]

data = get_data('training_data.txt')

np.random.seed(1337)

model = Sequential([
    LSTM(128, input_dim=1, return_sequences=False),
    Dense(128, 1),
    Activation('linear')
])

model.compile(loss='mean_squared_error', optimizer='rmsprop')
