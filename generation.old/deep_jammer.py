#!/usr/bin/env python
import argparse
import theano.tensor as T
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Permute, Lambda, Dropout, BatchNormalization
import piece_handler
import repository_handler
import data_parser

EPOCHS = 10000
BATCH_SIZE = 5

NUM_TIMESTEPS = 127
NUM_NOTES = 78
NUM_FEATURES = 80

TIME_MODEL_LAYER_1 = 300
TIME_MODEL_LAYER_2 = 300
NOTE_MODEL_LAYER_1 = 100
NOTE_MODEL_LAYER_2 = 50
OUTPUT_LAYER = 2

PIECE_LENGTH = 200
DROPOUT_PROBABILITY = 0.5

EPSILON = np.spacing(np.float32(1.0))

ARE_CHECKPOINTS_ENABLED = True
CHECKPOINT_DIRECTORY = 'checkpoints'
CHECKPOINT_THRESHOLD = 50

ARE_ACCURACIES_SAVED = True
ACCURACIES_DIRECTORY = 'accuracies'
ACCURACIES_THRESHOLD = 50
ACCURACY_DECAY_RATE = 0.9

IS_GPU_USED = False 


def train(model, pieces):
    loss_history = []
    moving_average_loss = 0.0

    for epoch in xrange(EPOCHS):
        print 'Training on epoch %s/%s...' % (epoch, EPOCHS)

        X, y = piece_handler.get_piece_batch(pieces, BATCH_SIZE)

        loss, _ = model.train_on_batch(X, y)
            
        moving_average_loss = moving_average_loss * ACCURACY_DECAY_RATE + loss * (1 - ACCURACY_DECAY_RATE)
        loss_history.append(moving_average_loss)

        print 'Loss =', loss

        if ARE_CHECKPOINTS_ENABLED and epoch % CHECKPOINT_THRESHOLD == 0:
            filename = '%s/model-weights-%s.h5' % (CHECKPOINT_DIRECTORY, epoch)
            model.save_weights(filename)

        if ARE_ACCURACIES_SAVED and epoch % ACCURACIES_THRESHOLD == 0:
            if IS_GPU_USED:
                filename = '%s/model-accuracies.txt' % ACCURACIES_DIRECTORY

                f = open(filename, 'w')
                for moving_average_loss in loss_history:
                    f.write('%s\n' % moving_average_loss)
            else:
                import matplotlib.pyplot as plt

                filename = '%s/model-accuracies.png' % ACCURACIES_DIRECTORY

                figure = plt.figure()
                plt.plot(loss_history)
                figure.suptitle('Loss Analysis')
                plt.xlabel('Iterations')
                plt.ylabel('Loss')

                figure.savefig(filename)


def compose_piece(model, start_note):
    inputs = [start_note]
    outputs = []

    for i in xrange(PIECE_LENGTH):
        X_in = inputs[i]
        y_pred = model.predict(X_in, batch_size=1).reshape((78, 2))

        rand_mask = np.random.uniform(size=y_pred.shape)
        y_pred = (rand_mask < y_pred)
        y_pred[:, 1] *= y_pred[:, 0]

        input = np.array(data_parser.get_single_input_form(y_pred, i)).reshape((1, 78, 80))

        inputs.append(input)
        outputs.append(y_pred)

    return np.asarray(outputs)


# def objective(y_true, y_pred):
#     played_likelihoods = T.sum(T.log(2 * y_pred[:, :, :, 0] * y_true[:, :, :, 0] - y_pred[:, :, :, 0] - y_true[:, :, :, 0] + 1 + EPSILON))
#
#     mask = y_true[:, :, :, 0]
#     articulated_likelihoods = T.sum(mask * (T.log(2 * y_pred[:, :, :, 1] * y_true[:, :, :, 1] - y_pred[:, :, :, 1] - y_true[:, :, :, 1] + 1 + EPSILON)))
#
#     return T.neg(played_likelihoods + articulated_likelihoods)

def objective(y_true, y_pred):
    active_notes = T.shape_padright(y_true[:, :, :, 0])
    mask = T.concatenate([T.ones_like(active_notes), active_notes], axis=3)

    log_likelihoods = mask * T.log(2 * y_pred * y_true - y_pred - y_true + 1 + EPSILON)

    return T.neg(T.sum(log_likelihoods))


def get_training_model():
    add_dimension_1 = lambda x: x.reshape([1, BATCH_SIZE, NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES])
    get_expanded_shape_1 = lambda shape: [1, BATCH_SIZE, NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES]
    remove_dimension_1 = lambda x: x.reshape([BATCH_SIZE * NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES])
    get_contracted_shape_1 = lambda shape: [BATCH_SIZE * NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES]

    add_dimension_2 = lambda x: x.reshape([1, BATCH_SIZE, NUM_NOTES, NUM_TIMESTEPS, TIME_MODEL_LAYER_2])
    get_expanded_shape_2 = lambda shape: [1, BATCH_SIZE, NUM_NOTES, NUM_TIMESTEPS, TIME_MODEL_LAYER_2]
    remove_dimension_2 = lambda x: x.reshape([BATCH_SIZE * NUM_TIMESTEPS, NUM_NOTES, TIME_MODEL_LAYER_2])
    get_contracted_shape_2 = lambda shape: [BATCH_SIZE * NUM_TIMESTEPS, NUM_NOTES, TIME_MODEL_LAYER_2]

    reshape_1 = lambda x: x.reshape([BATCH_SIZE, NUM_TIMESTEPS, NUM_NOTES, OUTPUT_LAYER])
    get_reshape_shape_1 = lambda shape: [BATCH_SIZE, NUM_TIMESTEPS, NUM_NOTES, OUTPUT_LAYER]


    return Sequential([
        Lambda(add_dimension_1, output_shape=get_expanded_shape_1, batch_input_shape=(BATCH_SIZE, NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES)),
        Permute((1, 3, 2, 4)),
        Lambda(remove_dimension_1, output_shape=get_contracted_shape_1),

        LSTM(TIME_MODEL_LAYER_1, return_sequences=True),
        Dropout(DROPOUT_PROBABILITY),
        LSTM(TIME_MODEL_LAYER_2, return_sequences=True),
        Dropout(DROPOUT_PROBABILITY),

        Lambda(add_dimension_2, output_shape=get_expanded_shape_2),
        Permute((1, 3, 2, 4)),
        Lambda(remove_dimension_2, output_shape=get_contracted_shape_2),

        LSTM(NOTE_MODEL_LAYER_1, return_sequences=True),
        Dropout(DROPOUT_PROBABILITY),
        LSTM(NOTE_MODEL_LAYER_2, return_sequences=True),
        Dropout(DROPOUT_PROBABILITY),

        TimeDistributed(Dense(OUTPUT_LAYER)),
        Dropout(DROPOUT_PROBABILITY),

        Activation('sigmoid'),

        Lambda(reshape_1, output_shape=get_reshape_shape_1)
    ])


def get_composition_model():
    unbroadcast = lambda x: T.unbroadcast(x, 0)
    get_shape = lambda x: x

    add_dimension_1 = lambda x: x.reshape([1, 1, NUM_NOTES, NUM_FEATURES])
    get_expanded_shape_1 = lambda shape: [1, 1, NUM_NOTES, NUM_FEATURES]
    remove_dimension_1 = lambda x: x.reshape([NUM_NOTES, 1, NUM_FEATURES])
    get_contracted_shape_1 = lambda shape: [NUM_NOTES, 1, NUM_FEATURES]

    add_dimension_2 = lambda x: x.reshape([1, NUM_NOTES, 1, TIME_MODEL_LAYER_2])
    get_expanded_shape_2 = lambda shape: [1, NUM_NOTES, 1, TIME_MODEL_LAYER_2]
    remove_dimension_2 = lambda x: x.reshape([1, NUM_NOTES, TIME_MODEL_LAYER_2])
    get_contracted_shape_2 = lambda shape: [1, NUM_NOTES, TIME_MODEL_LAYER_2]

    reshape_1 = lambda x: x.reshape([1, 1, NUM_NOTES, OUTPUT_LAYER])
    get_reshape_shape_1 = lambda shape: [1, 1, NUM_NOTES, OUTPUT_LAYER]

    return Sequential([
        Lambda(add_dimension_1, output_shape=get_expanded_shape_1, input_shape=(NUM_NOTES, NUM_FEATURES)),
        Permute((2, 1, 3)),
        Lambda(remove_dimension_1, output_shape=get_contracted_shape_1),

        LSTM(TIME_MODEL_LAYER_1, return_sequences=True),
        Dropout(DROPOUT_PROBABILITY),
        LSTM(TIME_MODEL_LAYER_2, return_sequences=True),
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

        Activation('sigmoid'),

        Lambda(reshape_1, output_shape=get_reshape_shape_1)
    ])


def main():
    print 'Generating the training model...'
    training_model = get_training_model()

    print 'Compiling the training model...'
    # Note that these settings are based on the post (which uses lr=0.01 and eps=1e-6)
    optimizer = Adadelta(lr=0.01, epsilon=1e-6)
    training_model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

    print 'Retrieving the repository...'
    pieces = repository_handler.load_repository(args.repository)

    print 'Learning...'
    train(training_model, pieces)

    print 'Retrieving the weights...'
    weights = training_model.get_weights()

    print 'Generating the composition model...'
    composition_model = get_composition_model()

    print 'Compiling the composition model...'
    composition_model.compile(loss=objective, optimizer=optimizer)

    print 'Setting the weights...'
    composition_model.set_weights(weights)

    print 'Composing a piece...'
    random_batch, _ = piece_handler.get_piece_batch(pieces, 5)
    initial_note = random_batch[0][0].reshape((1, 78, 80))
    piece = compose_piece(composition_model, initial_note)
    piece_handler.save_piece(piece, args.piece)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a creative, ingenious, classical piece.')
    parser.add_argument('piece', metavar='piece', help='the name of the new piece')
    parser.add_argument('repository', metavar='repository', help='the name of the repository')
    args = parser.parse_args()

    main()
