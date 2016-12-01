#!/usr/bin/env python
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Reshape, Permute, Lambda, Dropout
import numpy as np
import multi_training

NUM_EPOCHS = 10
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

DROPOUT_PROBABILITY = 0.5

def generate_dataset(pieces, size):
    X = []
    y = []

    for _ in xrange(size):
        segment = multi_training.get_piece_segment(pieces)
        X.append(segment[0])
        y.append(segment[1])

    X = np.array([np.array(X)]).reshape((size, NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES))
    y = np.array([np.array(y)]).reshape((size, NUM_TIMESTEPS, NUM_NOTES, OUTPUT_LAYER))

    return X, y

add_dimension_1 = lambda x: x.reshape([1, 1,  NUM_NOTES, NUM_FEATURES])
get_expanded_shape_1 = lambda shape: [1, 1, NUM_NOTES, NUM_FEATURES]
remove_dimension_1 = lambda x: x.reshape([NUM_NOTES, 1, NUM_FEATURES])
get_contracted_shape_1 = lambda shape: [NUM_NOTES, 1, NUM_FEATURES]

add_dimension_2 = lambda x: x.reshape([1, NUM_NOTES, 1, TIME_MODEL_LAYER_2])
get_expanded_shape_2 = lambda shape: [1, NUM_NOTES, 1, TIME_MODEL_LAYER_2]
remove_dimension_2 = lambda x: x.reshape([1, NUM_NOTES, TIME_MODEL_LAYER_2])
get_contracted_shape_2 = lambda shape: [1, NUM_NOTES, TIME_MODEL_LAYER_2]

remove_dimension_3 = lambda x: x.reshape([NUM_NOTES, NOTE_MODEL_LAYER_2])
get_contracted_shape_3 = lambda shape: [NUM_NOTES, NOTE_MODEL_LAYER_2]
add_dimension_3 = lambda x: x.reshape([1, NUM_NOTES, OUTPUT_LAYER])
get_expanded_shape_3 = lambda shape: [1, NUM_NOTES, OUTPUT_LAYER]

def main():
    model = Sequential([
        Lambda(add_dimension_1, output_shape=get_expanded_shape_1, input_shape=(NUM_NOTES, NUM_FEATURES)),
        Permute((2, 1, 3)),
        Lambda(remove_dimension_1, output_shape=get_contracted_shape_1),

        LSTM(TIME_MODEL_LAYER_1, return_sequences=True, stateful=True),
        # Dropout(DROPOUT_PROBABILITY),
        LSTM(TIME_MODEL_LAYER_2, return_sequences=True, stateful=True),
        # Dropout(DROPOUT_PROBABILITY),

        Lambda(add_dimension_2, output_shape=get_expanded_shape_2),
        Permute((2, 1, 3)),
        Lambda(remove_dimension_2, output_shape=get_contracted_shape_2),

        LSTM(NOTE_MODEL_LAYER_1, return_sequences=True),
        # Dropout(DROPOUT_PROBABILITY),
        LSTM(NOTE_MODEL_LAYER_2, return_sequences=True),
        # Dropout(DROPOUT_PROBABILITY),

        Lambda(remove_dimension_3, output_shape=get_contracted_shape_3),

        Dense(OUTPUT_LAYER),

        Lambda(add_dimension_3, output_shape=get_expanded_shape_3),

        Activation('sigmoid')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    pieces = multi_training.load_pieces('music')


    print "Generating training set..."
    X_train, y_train = generate_dataset(pieces, NUM_SEGMENTS)

    print X_train.shape

    print "Generating test set..."
    X_test, y_test = generate_dataset(pieces, NUM_TESTS)

    print X_test.shape

    print "Training the model..."

    for _ in xrange(NUM_EPOCHS):
        for i in xrange(NUM_SEGMENTS):
            print 'Training on batch %s/%s' % (i, NUM_SEGMENTS * NUM_EPOCHS)

            for j in xrange(NUM_TIMESTEPS):

                X = np.expand_dims(X_train[i, j], axis=0)
                y = np.expand_dims(y_train[i, j], axis=0)

                model.train_on_batch(X, y)
            model.reset_states()

    # print "Testing the model..."
    # for _ in xrange(NUM_TESTS):
    #
    #
    #     start = width * i
    #     end = width * (i + 1)
    #
    #     X = X_test[:, start:end, :]
    #     y = y_test[:, start:end, :]
    #
    #     print model.test_on_batch(X, y)
    #     model.reset_states
    #
    #
    # for i in xrange(NUM_GEN):
    #     # TODO Initialize song segments
    #
    #     X_gen = np.rand((NUM_TIMESTEPS * NUM_NOTES, NUM_FEATURES))
    #     model.reset_states()
    #
    #     X_in = X_gen[0,:]
    #
    #     for i in xrange(NUM_TIMESTEPS):
    #
    #         start = NUM_NOTES * i
    #         end = NUM_NOTES * (i + 1)
    #
    #         X_in = model.predict(X_in, batch_size=NUM_NOTES, verbose=0)
    #         X_gen[start:end, :] = X_in


if __name__ == '__main__':
    main()
