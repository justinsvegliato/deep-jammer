#!/usr/bin/env python
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Reshape, Permute, Lambda

NUM_SEGMENTS = 10
NUM_NOTES = 10
NUM_TIMESTEPS = 10
NUM_FEATURES = 10

TIME_MODEL_LAYER_1 = 300
TIME_MODEL_LAYER_2 = 300

NOTE_MODEL_LAYER_1 = 100
NOTE_MODEL_LAYER_2 = 50

OUTPUT_LAYER = 2


def add_dimension(x):
    return x

def get_expanded_shape(shape):
    return (1,) + shape

def remove_dimension(x):
    return x

def get_squeezed_shape(shape):
    return shape[1:]

def main():
    model = Sequential([
        Lambda(add_dimension, output_shape=get_expanded_shape, input_shape=(NUM_SEGMENTS, NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES)),
        Reshape((NUM_SEGMENTS * NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES)),
        Lambda(remove_dimension, output_shape=get_squeezed_shape),
        LSTM(TIME_MODEL_LAYER_1, return_sequences=True),
        LSTM(TIME_MODEL_LAYER_2, return_sequences=True),
        Lambda(add_dimension, output_shape=get_expanded_shape),
        Reshape((NUM_SEGMENTS, NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES)),
        Permute((1, 3, 2, 4)),
        Reshape((NUM_SEGMENTS * NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES)),
        Lambda(remove_dimension, output_shape=get_squeezed_shape),
        LSTM(NOTE_MODEL_LAYER_1, return_sequences=True),
        LSTM(NOTE_MODEL_LAYER_2, return_sequences=True),
        Lambda(add_dimension, output_shape=get_expanded_shape),
        Reshape((NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES)),
        Lambda(remove_dimension, output_shape=get_squeezed_shape),
        Dense(OUTPUT_LAYER),
        Activation('sigmoid')
    ])

if __name__ == '__main__':
    main()

