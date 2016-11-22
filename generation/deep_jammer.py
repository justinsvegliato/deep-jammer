#!/usr/bin/env python
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Nadam

BATCH_SIZE = 10
TIMESTEPS = 10
INPUT_NOTE_COUNT = 80

def main():
    training_examples = []
    training_labels = []

    model = Sequential([
        LSTM(300, batch_input_shape=(BATCH_SIZE, TIMESTEPS, INPUT_NOTE_COUNT)),
        LSTM(300),
        LSTM(100),
        LSTM(50)
    ])

    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])

    model.fit(training_examples, training_labels)

if __name__ == '__main__':
    main()
