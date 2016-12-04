#!/usr/bin/env python
import argparse
import theano as T
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Permute, Lambda, Dropout, BatchNormalization
import piece_handler
import repository_handler
import data_parser

NUM_EPOCHS = 5
NUM_TESTS = 10

NUM_SEGMENTS = 2 
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
CHECKPOINT_THRESHOLD = 100

ARE_STORING_ACCURACIES = True
ACCURACIES_DIRECTORY = 'accuracies'
ACCURACIES_THRESHOLD = 25
ACCURACY_DECAY_RATE = 0.9


def train(model, X_train, y_train):

    loss_history = []
    moving_average_loss = 0.0

    for epoch in xrange(NUM_EPOCHS):
        for segment in xrange(NUM_SEGMENTS):
            id = segment + epoch * NUM_SEGMENTS + 1

            print 'Training on batch %s/%s...' % (id, NUM_SEGMENTS * NUM_EPOCHS)

            X = X_train[segment]
            y = y_train[segment]

            loss, _ = model.train_on_batch(X, y)
            
            moving_average_loss = moving_average_loss * ACCURACY_DECAY_RATE + loss * (1 - ACCURACY_DECAY_RATE)
            loss_history.append(moving_average_loss)

            print 'Loss:', loss
                    
            if ARE_CHECKPOINTS_ENABLED and id % CHECKPOINT_THRESHOLD == 0:
                filename = '%s/model-weights-%s.h5' % (CHECKPOINT_DIRECTORY, id)
                model.save_weights(filename)

            if ARE_STORING_ACCURACIES and id % ACCURACIES_THRESHOLD == 0:
                figure = plt.figure()
                plt.plot(loss_history)
                figure.suptitle('Loss Analysis')
                plt.xlabel('Iterations')
                plt.ylabel('Loss')

                filename = '%s/model-accuracies.png' % ACCURACIES_DIRECTORY
                figure.savefig(filename)

            model.reset_states()


def test(model, X_test, y_test):
    for test in xrange(NUM_TESTS):
        print 'Testing on batch %s/%s...' % (test + 1, NUM_TESTS * NUM_EPOCHS)

        for timestep in xrange(NUM_TIMESTEPS):
            X = np.expand_dims(X_test[test, timestep], axis=0)
            y = np.expand_dims(y_test[test, timestep], axis=0)
            print '(Loss, Accuracy):', model.test_on_batch(X, y)

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


def objective(y_true, y_pred):
    epsilon = 1.0e-5
    
    #y_true = y_true.reshape((NUM_TIMESTEPS, NUM_NOTES, OUTPUT_LAYER))
    #y_pred = y_pred.reshape((NUM_TIMESTEPS, NUM_NOTES, OUTPUT_LAYER))

    mask = y_true[:, :, 0]

    played_likelihoods = T.tensor.sum(T.tensor.log(2 * y_pred[:, :, 0] * y_true[:, :, 0] - y_pred[:, :, 0] - y_true[:, :, 0] + 1 + epsilon))
    articulated_likelihoods = T.tensor.sum(mask * (T.tensor.log(2 * y_pred[:, :, 1] * y_true[:, :, 1] - y_pred[:, :, 1] - y_true[:, :, 1] + 1 + epsilon)))

    return T.tensor.neg(played_likelihoods + articulated_likelihoods)


unbroadcast = lambda x: T.tensor.unbroadcast(x, 0)
get_shape = lambda x: x

add_dimension_1 = lambda x: x.reshape([1, NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES])
get_expanded_shape_1 = lambda shape: [1, NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES]
remove_dimension_1 = lambda x: x.reshape([NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES])
get_contracted_shape_1 = lambda shape: [NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES]

add_dimension_2 = lambda x: x.reshape([1, NUM_NOTES, NUM_TIMESTEPS, TIME_MODEL_LAYER_2])
get_expanded_shape_2 = lambda shape: [1, NUM_NOTES, NUM_TIMESTEPS, TIME_MODEL_LAYER_2]
remove_dimension_2 = lambda x: x.reshape([NUM_TIMESTEPS, NUM_NOTES, TIME_MODEL_LAYER_2])
get_contracted_shape_2 = lambda shape: [NUM_TIMESTEPS, NUM_NOTES, TIME_MODEL_LAYER_2]


def main():
    model = Sequential([
        Lambda(add_dimension_1, output_shape=get_expanded_shape_1, batch_input_shape=(NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES)),
        Permute((2, 1, 3)),
        Lambda(remove_dimension_1, output_shape=get_contracted_shape_1),

        LSTM(TIME_MODEL_LAYER_1, return_sequences=True),
        # #Dropout(DROPOUT_PROBABILITY),
        # BatchNormalization(),
        LSTM(TIME_MODEL_LAYER_2, return_sequences=True),
        # #Dropout(DROPOUT_PROBABILITY),
        # BatchNormalization(),

        Lambda(add_dimension_2, output_shape=get_expanded_shape_2),
        Permute((2, 1, 3)),
        Lambda(remove_dimension_2, output_shape=get_contracted_shape_2),

        LSTM(NOTE_MODEL_LAYER_1, return_sequences=True),
        # #Dropout(DROPOUT_PROBABILITY),
        # BatchNormalization(),
        LSTM(NOTE_MODEL_LAYER_2, return_sequences=True),
        # #Dropout(DROPOUT_PROBABILITY),
        # BatchNormalization(),


        TimeDistributed(Dense(OUTPUT_LAYER)),
        # BatchNormalization(),

        Activation('sigmoid')
    ])
    optimizer = Adadelta(lr=0.01, epsilon=1e-6)
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

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

