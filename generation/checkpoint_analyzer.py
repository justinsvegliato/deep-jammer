#!/usr/bin/env python
import theano as T
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Permute, Lambda, Dropout
from keras.optimizers import Adadelta
import deep_jammer
import repository_handler
import piece_handler

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

DROPOUT_PROBABILITY = 0.5

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

    optimizer = Adadelta(lr=0.01, rho=0.95, epsilon=1e-06)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.load_weights('checkpoints/example-model-weights.h5')

    print 'Generating the initial note of the piece...'
    repository = repository_handler.load_repository('example-repository')
    X_train, _ = piece_handler.get_dataset(repository, 5)
    initial_note = X_train[0][0].reshape((1, 78, 80))

    print 'Generating a piece...'
    piece = deep_jammer.compose_piece(model, initial_note)
    piece_handler.save_piece(piece, 'test')


if __name__ == '__main__':
    main()
