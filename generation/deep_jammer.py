#!/usr/bin/env python
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Reshape, Permute, Lambda, Dropout
import numpy as np
import multi_training

NUM_EPOCHS = 10
NUM_TRAIN = 100
NUM_TEST = 10

NUM_SEGMENTS = 2
NUM_TIMESTEPS = 128
NUM_NOTES = 78
NUM_FEATURES = 80

TIME_MODEL_LAYER_1 = 300
TIME_MODEL_LAYER_2 = 300
NOTE_MODEL_LAYER_1 = 100
NOTE_MODEL_LAYER_2 = 50
OUTPUT_LAYER = 2

DROPOUT_PROBABILITY = 0.5

def generate_dataset(pieces, size):
    X_train = []
    y_train = []

    for _ in xrange(size):
        segment = multi_training.get_piece_segment(pieces)
        X.append(segment[0])
        y.append(segment[1])

    X = np.array([np.array(X_train)]).reshape((1, size * NUM_TIMESTEPS * NUM_NOTES, NUM_FEATURES))
    y = np.array([np.array(y_train)]).reshape((1, size * NUM_TIMESTEPS * NUM_NOTES, NUM_FEATURES))

    return X, y

add_dimension_1 = lambda x: x.reshape([1, NUM_TIMESTEPS * NUM_NOTES, NUM_FEATURES])
get_expanded_shape_1 = lambda shape: [1, NUM_TIMESTEPS * NUM_NOTES, NUM_FEATURES]
remove_dimension_1 = lambda x: x.reshape([NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES])
get_contracted_shape_1 = lambda shape: [NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES]

add_dimension_2 = lambda x: x.reshape([1, NUM_NOTES, NUM_TIMESTEPS, TIME_MODEL_LAYER_2])
get_expanded_shape_2 = lambda shape: [1, NUM_NOTES, NUM_TIMESTEPS, TIME_MODEL_LAYER_2]
remove_dimension_2 = lambda x: x.reshape([NUM_TIMESTEPS, NUM_NOTES, TIME_MODEL_LAYER_2])
get_contracted_shape_2 = lambda shape: [NUM_TIMESTEPS, NUM_NOTES, TIME_MODEL_LAYER_2]

add_dimension_3 = lambda x: x.reshape([1, NUM_TIMESTEPS, NUM_NOTES, NOTE_MODEL_LAYER_2])
get_expanded_shape_3 = lambda shape: [1, NUM_TIMESTEPS, NUM_NOTES, NOTE_MODEL_LAYER_2]
remove_dimension_3 = lambda x: x.reshape([NUM_TIMESTEPS * NUM_NOTES, NOTE_MODEL_LAYER_2])
get_contracted_shape_3 = lambda shape: [NUM_TIMESTEPS * NUM_NOTES, NOTE_MODEL_LAYER_2]

add_dimension_4 = lambda x: x.reshape([1, NUM_TIMESTEPS * NUM_NOTES, OUTPUT_LAYER])
get_expanded_shape_4 = lambda shape: [1, NUM_TIMESTEPS * NUM_NOTES, OUTPUT_LAYER]

def main():
    model = Sequential([
        Lambda(add_dimension_1, output_shape=get_expanded_shape_1, input_shape=(NUM_TIMESTEPS * NUM_NOTES, NUM_FEATURES)),
        Reshape((1, NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES)),
        Permute((1, 3, 2, 4)),
        Reshape((NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES)),
        Lambda(remove_dimension_1, output_shape=get_contracted_shape_1),

        LSTM(TIME_MODEL_LAYER_1, return_sequences=True),
        # Dropout(DROPOUT_PROBABILITY),
        LSTM(TIME_MODEL_LAYER_2, return_sequences=True),
        # Dropout(DROPOUT_PROBABILITY),

        Lambda(add_dimension_2, output_shape=get_expanded_shape_2),
        Reshape((1, NUM_NOTES, NUM_TIMESTEPS, TIME_MODEL_LAYER_2)),
        Permute((1, 3, 2, 4)),
        Reshape((NUM_TIMESTEPS, NUM_NOTES, TIME_MODEL_LAYER_2)),
        Lambda(remove_dimension_2, output_shape=get_contracted_shape_2),

        LSTM(NOTE_MODEL_LAYER_1, return_sequences=True),
        # Dropout(DROPOUT_PROBABILITY),
        LSTM(NOTE_MODEL_LAYER_2, return_sequences=True),
        # Dropout(DROPOUT_PROBABILITY),

        Lambda(add_dimension_3, output_shape=get_expanded_shape_3),
        Reshape((NUM_TIMESTEPS * NUM_NOTES, NOTE_MODEL_LAYER_2)),
        Lambda(remove_dimension_3, output_shape=get_contracted_shape_3),

        Dense(OUTPUT_LAYER),
        Lambda(add_dimension_4, output_shape=get_expanded_shape_4),
        Activation('sigmoid')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    pieces = multi_training.load_pieces('music')

    X_train, y_train = generate_dataset(pieces, NUM_SEGMENTS)
    X_test, y_test = generate_dataset(pieces, NUM_TEST)

    # X_test = np.array([np.array([piece_segment_1[0], piece_segment_2[0]])])
    # X_test = np.reshape(X_train, (1, NUM_SEGMENTS * NUM_TIMESTEPS * NUM_NOTES, NUM_FEATURES))
    # for i in xrange(NUM_TEST):
    #     piece_segment_1 = multi_training.get_piece_segment(pieces)

    # X_train = np.array([piece_segment_1[0], piece_segment_2[0]])
    # X_train = np.array([np.array([piece_segment_1[0], piece_segment_2[0]])])
    # X_train = np.reshape(X_train, (1, NUM_SEGMENTS * NUM_TIMESTEPS * NUM_NOTES, NUM_FEATURES))
    # print X_train.shape
    #
    # y_train = np.array([np.array([piece_segment_1[1], piece_segment_2[1]])])
    # y_train = np.reshape(y_train, (1, NUM_SEGMENTS * NUM_TIMESTEPS * NUM_NOTES, OUTPUT_LAYER))
    # print y_train.shape
    #
    # X_test = np.

    width = NUM_TIMESTEPS * NUM_NOTES
    for i in xrange(NUM_SEGMENTS * NUM_EPOCHS):
        print 'Training on batch %s/%s' % (i, NUM_SEGMENTS * NUM_EPOCHS)

        segment = i % NUM_SEGMENTS
        start = width * segment
        # TODO: Check this shit out for -1
        end = width * (segment + 1)

        X = X_train[:, start:end, :]
        y = y_train[:, start:end, :]

        model.train_on_batch(X, y)

    # model.fit(X_train, y_train, nb_epoch=NUM_EPOCHS, batch_size=NUM_TIMESTEPS * NUM_NOTES)

    for i in xrange(NUM_TEST):
        start = width * i
        # TODO: Check this shit out for -1
        end = width * (i + 1)

        X = X_test[:, start:end, :]
        y = y_test[:, start:end, :]

        print model.test_on_batch(X, y)

if __name__ == '__main__':
    main()
