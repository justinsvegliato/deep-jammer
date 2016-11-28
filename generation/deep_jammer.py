#!/usr/bin/env python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Reshape, Permute

NUM_SEGMENTS = 10
NUM_NOTES = 10
NUM_TIMESTEPS = 10
NUM_FEATURES = 10

TIME_MODEL_LAYER_1 = 300
TIME_MODEL_LAYER_2 = 300

NOTE_MODEL_LAYER_1 = 100
NOTE_MODEL_LAYER_2 = 50

OUTPUT_LAYER = 2

def main():
    model = Sequential([
        # Lambda Add Dimension , input_shape=(NUM_SEGMENTS, NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES)
        Reshape((NUM_SEGMENTS * NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES)),
        # Lambda Remove Dimension
        LSTM(TIME_MODEL_LAYER_1, return_sequences=True),
        LSTM(TIME_MODEL_LAYER_2, return_sequences=True),
        # Lambda Add Dimension
        Reshape((NUM_SEGMENTS, NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES)),
        Permute((1, 3, 2, 4)),
        Reshape((NUM_SEGMENTS * NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES)),
        # Lambda Remove Dimension
        LSTM(NOTE_MODEL_LAYER_1, return_sequences=True),
        LSTM(NOTE_MODEL_LAYER_2, return_sequences=True),
        # Lambda Add Dimension
        Reshape((NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES)),
        # Lambda Remove Dimension
        Dense(OUTPUT_LAYER),
        Activation('sigmoid')
    ])

if __name__ == '__main__':
    main()

