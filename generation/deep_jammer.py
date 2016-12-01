#!/usr/bin/env python
import theano as T
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Permute, Lambda, Dropout
import multi_training

NUM_EPOCHS = 2
NUM_TESTS = 2
NUM_GEN = 10
NUM_SEGMENTS = 2
NUM_TIMESTEPS = 128
NUM_NOTES = 78
NUM_FEATURES = 80

TIME_MODEL_LAYER_1 = 300
TIME_MODEL_LAYER_2 = 300
NOTE_MODEL_LAYER_1 = 100
NOTE_MODEL_LAYER_2 = 50
OUTPUT_LAYER = 2

def generate_dataset(pieces, size):
    X = []
    y = []

    for _ in xrange(size):
        segment = multi_training.get_piece_segment(pieces)
        X.append(segment[0])
        y.append(segment[1])

    return np.array(X), np.array(y)

unbroadcast = lambda x: T.tensor.unbroadcast(x, 0)
get_shape = lambda x: x

add_dimension_1 = lambda x: x.reshape([1, 1, NUM_NOTES, NUM_FEATURES])
get_expanded_shape_1 = lambda shape: [1, 1, NUM_NOTES, NUM_FEATURES]
remove_dimension_1 = lambda x: x.reshape([NUM_NOTES, 1, NUM_FEATURES])
get_contracted_shape_1 = lambda shape: [NUM_NOTES, 1, NUM_FEATURES]

add_dimension_2 = lambda x: x.reshape([1, NUM_NOTES, 1, TIME_MODEL_LAYER_2])
get_expanded_shape_2 = lambda shape: [1, NUM_NOTES, 1, TIME_MODEL_LAYER_2]
remove_dimension_2 = lambda x: x.reshape([1, NUM_NOTES, TIME_MODEL_LAYER_2])
get_contracted_shape_2 = lambda shape: [1, NUM_NOTES, TIME_MODEL_LAYER_2]

def main():
    model = Sequential([
        Lambda(add_dimension_1, output_shape=get_expanded_shape_1, input_shape=(NUM_NOTES, NUM_FEATURES)),
        Permute((2, 1, 3)),
        Lambda(remove_dimension_1, output_shape=get_contracted_shape_1),

        LSTM(TIME_MODEL_LAYER_1, return_sequences=True, stateful=True),
        LSTM(TIME_MODEL_LAYER_2, return_sequences=True, stateful=True),

        Lambda(add_dimension_2, output_shape=get_expanded_shape_2),
        Permute((2, 1, 3)),
        Lambda(remove_dimension_2, output_shape=get_contracted_shape_2),

        Lambda(unbroadcast, output_shape=get_shape),
        LSTM(NOTE_MODEL_LAYER_1, return_sequences=True),
        LSTM(NOTE_MODEL_LAYER_2, return_sequences=True),

        TimeDistributed(Dense(OUTPUT_LAYER)),
        Activation('sigmoid')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    print 'Retrieving pieces...'
    pieces = multi_training.load_pieces('music')

    print 'Generating training set...'
    X_train, y_train = generate_dataset(pieces, NUM_SEGMENTS)

    print 'Generating test set...'
    X_test, y_test = generate_dataset(pieces, NUM_TESTS)

    print 'Training the model...'
    for epoch in xrange(NUM_EPOCHS):
        for segment in xrange(NUM_SEGMENTS):
            print 'Training on batch %s/%s...' % (segment + epoch * NUM_SEGMENTS + 1, NUM_SEGMENTS * NUM_EPOCHS)

            for timestep in xrange(NUM_TIMESTEPS):
                X = np.expand_dims(X_train[segment, timestep], axis=0)
                y = np.expand_dims(y_train[segment, timestep], axis=0)
                model.train_on_batch(X, y)

            model.reset_states()

    print 'Testing the model...'
    for segment in xrange(NUM_SEGMENTS):
        print 'Testing on batch %s/%s...' % (segment + 1, NUM_SEGMENTS * NUM_EPOCHS)

        for timestep in xrange(NUM_TIMESTEPS):
            X = np.expand_dims(X_test[segment, timestep], axis=0)
            y = np.expand_dims(y_test[segment, timestep], axis=0)
            model.test_on_batch(X, y)

        model.reset_states()

    generated_song = []
    for i in xrange(NUM_GEN):
        X_in = generated_song[i]
        y_pred = model.predict(X_in, batch_size=1)
        generated_song.append(y_pred)

if __name__ == '__main__':
    main()
