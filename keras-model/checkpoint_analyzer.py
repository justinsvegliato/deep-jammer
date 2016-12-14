#!/usr/bin/env python
import numpy as np
import theano as T
from keras.optimizers import Adam
import repository_handler
import piece_handler
import data_parser
import deep_jammer

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


def compose_piece(model, start_note):
    inputs = [start_note]
    outputs = []

    for i in xrange(PIECE_LENGTH):
        X_in = inputs[i]
        y_pred = model.predict(X_in, batch_size=1).reshape((78, 2))

        random_mask = np.random.uniform(size=y_pred.shape)
        y_pred[:, 0] = y_pred[:, 0] > random_mask[:, 0]
        y_pred[:, 1] = y_pred[:, 0] * (y_pred[:, 1] > random_mask[:, 1])

        input = np.array(data_parser.get_single_input_form(y_pred, i)).reshape((1, 78, 80))
        
        inputs.append(input)
        outputs.append(y_pred)

    return np.asarray(outputs)


def objective(y_true, y_pred):
    epsilon = 1.0e-5

    played_likelihoods = T.tensor.sum(T.tensor.log(2 * y_pred[:, :, 0] * y_true[:, :, 0] - y_pred[:, :, 0] - y_true[:, :, 0] + 1 + epsilon))

    mask = y_true[:, :, 0]
    articulated_likelihoods = T.tensor.sum(mask * (T.tensor.log(2 * y_pred[:, :, 1] * y_true[:, :, 1] - y_pred[:, :, 1] - y_true[:, :, 1] + 1 + epsilon)))

    return T.tensor.neg(played_likelihoods + articulated_likelihoods)


def main():
    print 'Generating the composition model...'
    composition_model = deep_jammer.get_composition_model()

    print 'Compiling the composition model...'
    optimizer = Adam(lr=1)
    composition_model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

    print 'Loading the weights...'
    composition_model.load_weights('checkpoints/model-weights-40.h5')

    print 'Generating the initial note of the piece...'
    repository = repository_handler.load_repository('1-repository')
    X_train, _ = piece_handler.get_piece_batch(repository, 5)
    initial_note = X_train[0][0].reshape((1, 78, 80))

    print 'Generating a piece...'
    piece = compose_piece(composition_model, initial_note)
    piece_handler.save_piece(piece, 'checkpoint-piece')


if __name__ == '__main__':
    main()
