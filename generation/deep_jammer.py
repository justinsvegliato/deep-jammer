#!/usr/bin/env python
import argparse
import theano as T
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Permute, Lambda, Dropout
import piece_handler
import repository_handler
import data_parser

NUM_EPOCHS = 10
NUM_TESTS = 10

NUM_SEGMENTS = 500
NUM_TIMESTEPS = 128
NUM_NOTES = 78
NUM_FEATURES = 80

TIME_MODEL_LAYER_1 = 300
TIME_MODEL_LAYER_2 = 300
NOTE_MODEL_LAYER_1 = 100
NOTE_MODEL_LAYER_2 = 50
OUTPUT_LAYER = 2

PIECE_LENGTH = 200
DROPOUT_PROBABILITY = 0.5

ARE_CHECKPOINTS_ENABLED = True
CHECKPOINT_DIRECTORY = 'checkpoints'
CHECKPOINT_THRESHOLD = 200


def train(model, X_train, y_train):
    for epoch in xrange(NUM_EPOCHS):
        for segment in xrange(NUM_SEGMENTS):
            id = segment + epoch * NUM_SEGMENTS + 1

            print 'Training on batch %s/%s...' % (id, NUM_SEGMENTS * NUM_EPOCHS)

            for timestep in xrange(NUM_TIMESTEPS):
                X = np.expand_dims(X_train[segment, timestep], axis=0)
                y = np.expand_dims(y_train[segment, timestep], axis=0)
                model.train_on_batch(X, y)

            if ARE_CHECKPOINTS_ENABLED and id % CHECKPOINT_THRESHOLD == 0:
                filename = '%s/model-weights-%s.h5' % (CHECKPOINT_DIRECTORY, id)
                model.save_weights(filename)

            model.reset_states()


def test(model, X_test, y_test):
    # TODO This function needs some work. It should iterate over NUM_TESTS, not NUM_SEGMENTS.
    for segment in xrange(NUM_SEGMENTS):
        print 'Testing on batch %s/%s...' % (segment + 1, NUM_SEGMENTS * NUM_EPOCHS)

        for timestep in xrange(NUM_TIMESTEPS):
            X = np.expand_dims(X_test[segment, timestep], axis=0)
            y = np.expand_dims(y_test[segment, timestep], axis=0)
            model.test_on_batch(X, y)

        model.reset_states()


def compose_piece(model, start_note):
    inputs = [start_note]
    outputs = []

    for i in xrange(PIECE_LENGTH):
        X_in = inputs[i]
        y_pred = model.predict(X_in, batch_size=1).reshape((78, 2))

        # Set the probabilities of the input to 0s and 1s through sampling
        rand_mask = np.random.uniform(size=y_pred.shape)
        y_pred = (rand_mask < y_pred)

        # Set articulate probabilities to 0 if the note is not played
        y_pred[:, 1] *= y_pred[:, 0]

        input = np.array(data_parser.get_single_input_form(y_pred, i)).reshape((1, 78, 80))

        inputs.append(input)
        outputs.append(y_pred)

    return np.asarray(outputs)


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
        Dropout(DROPOUT_PROBABILITY),
        LSTM(TIME_MODEL_LAYER_2, return_sequences=True, stateful=True),
        Dropout(DROPOUT_PROBABILITY),

        Lambda(add_dimension_2, output_shape=get_expanded_shape_2),
        Permute((2, 1, 3)),
        Lambda(remove_dimension_2, output_shape=get_contracted_shape_2),

        Lambda(unbroadcast, output_shape=get_shape),
        LSTM(NOTE_MODEL_LAYER_1, return_sequences=True),
        Dropout(DROPOUT_PROBABILITY),
        LSTM(NOTE_MODEL_LAYER_2, return_sequences=True),
        Dropout(DROPOUT_PROBABILITY),

        TimeDistributed(Dense(OUTPUT_LAYER)),
        Dropout(DROPOUT_PROBABILITY),

        Activation('sigmoid')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    print 'Retrieving repository...'
    repository = repository_handler.load_repository(args.repository)

    print 'Generating the training set...'
    X_train, y_train = piece_handler.get_dataset(repository, NUM_SEGMENTS)

    print 'Training the model...'
    train(model, X_train, y_train)

    # print 'Generating the test set...'
    # X_test, y_test = piece_handler.get_dataset(repository, NUM_TESTS)

    # print 'Testing the model...'
    # test(model, X_test, y_test)

    print 'Generating a piece...'
    # TODO Should the initial note be something else?
    initial_note = X_train[0][0].reshape((1, 78, 80))
    piece = compose_piece(model, initial_note)
    piece_handler.save_piece(piece, args.piece)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a creative, ingenious, classical piece.')
    parser.add_argument('piece', metavar='piece', help='the name of the new piece')
    parser.add_argument('repository', metavar='repository', help='the name of the repository')
    args = parser.parse_args()

    main()
