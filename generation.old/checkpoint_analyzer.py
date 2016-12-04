#!/usr/bin/env python
import numpy as np
import theano as T
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Permute, Lambda, Dropout
import deep_jammer
import repository_handler
import piece_handler
import data_parser

NUM_EPOCHS = 10
NUM_TESTS = 10

NUM_SEGMENTS = 100 
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
 
    cons = 1

    for i in xrange(PIECE_LENGTH):
        X_in = inputs[i]
        y_pred = model.predict(X_in, batch_size=1).reshape((78, 2))

        # Set the probabilities of the input to 0s and 1s through sampling
        random_mask = np.random.uniform(size=y_pred.shape)
        y_pred[:, 0] = (y_pred[:, 0] ** cons) > random_mask[:, 0]

        nnotes = np.sum(y_pred[:, 0])

        if nnotes < 2:
            if cons > 1:
                cons = 1

            cons -= 0.02
        else:
            cons += (1 - cons) * 0.3

        # Set articulate probabilities to 0 if the note is not played
        y_pred[:, 1] = y_pred[:, 0] * (y_pred[:, 1]  > random_mask[:, 1]) 

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
    model.load_weights('checkpoints/model-weights-3800.h5')

    print 'Generating the initial note of the piece...'
    repository = repository_handler.load_repository('200-repository')
    X_train, _ = piece_handler.get_dataset(repository, 5)
    initial_note = X_train[0][0].reshape((1, 78, 80))

    print 'Generating a piece...'
    piece = compose_piece(model, initial_note)
    piece_handler.save_piece(piece, 'samer')


if __name__ == '__main__':
    main()
