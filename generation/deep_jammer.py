#!/usr/bin/env python
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Reshape, Permute, Lambda, Dropout
import numpy as np
import multi_training

NUM_EPOCHS = 10
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

add_dimension_1 = lambda x: x.reshape([1, NUM_SEGMENTS, NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES])
get_expanded_shape_1 = lambda shape: [1, NUM_SEGMENTS, NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES]
remove_dimension_1 = lambda x: x.reshape([NUM_SEGMENTS * NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES])
get_contracted_shape_1 = lambda shape: [NUM_SEGMENTS * NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES]

add_dimension_2 = lambda x: x.reshape([1, NUM_SEGMENTS * NUM_NOTES, NUM_TIMESTEPS, TIME_MODEL_LAYER_2])
get_expanded_shape_2 = lambda shape: [1, NUM_SEGMENTS * NUM_NOTES, NUM_TIMESTEPS, TIME_MODEL_LAYER_2]
remove_dimension_2 = lambda x: x.reshape([NUM_SEGMENTS * NUM_TIMESTEPS, NUM_NOTES, TIME_MODEL_LAYER_2])
get_contracted_shape_2 = lambda shape: [NUM_SEGMENTS * NUM_TIMESTEPS, NUM_NOTES, TIME_MODEL_LAYER_2]

add_dimension_3 = lambda x: x.reshape([1, NUM_SEGMENTS * NUM_TIMESTEPS, NUM_NOTES, NOTE_MODEL_LAYER_2])
get_expanded_shape_3 = lambda shape: [1, NUM_SEGMENTS * NUM_TIMESTEPS, NUM_NOTES, NOTE_MODEL_LAYER_2]
remove_dimension_3 = lambda x: x.reshape([NUM_SEGMENTS * NUM_TIMESTEPS * NUM_NOTES, NOTE_MODEL_LAYER_2])
get_contracted_shape_3 = lambda shape: [NUM_SEGMENTS * NUM_TIMESTEPS * NUM_NOTES, NOTE_MODEL_LAYER_2]

def main():
    model = Sequential([
        Lambda(add_dimension_1, output_shape=get_expanded_shape_1, input_shape=(NUM_SEGMENTS, NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES)),
        Permute((1, 3, 2, 4)),
        Reshape((NUM_SEGMENTS * NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES)),
        Lambda(remove_dimension_1, output_shape=get_contracted_shape_1),

        LSTM(TIME_MODEL_LAYER_1, return_sequences=True),
        # Dropout(DROPOUT_PROBABILITY),
        LSTM(TIME_MODEL_LAYER_2, return_sequences=True),
        # Dropout(DROPOUT_PROBABILITY),

        Lambda(add_dimension_2, output_shape=get_expanded_shape_2),
        Reshape((NUM_SEGMENTS, NUM_NOTES, NUM_TIMESTEPS, TIME_MODEL_LAYER_2)),
        Permute((1, 3, 2, 4)),
        Reshape((NUM_SEGMENTS * NUM_TIMESTEPS, NUM_NOTES, TIME_MODEL_LAYER_2)),
        Lambda(remove_dimension_2, output_shape=get_contracted_shape_2),

        LSTM(NOTE_MODEL_LAYER_1, return_sequences=True),
        # Dropout(DROPOUT_PROBABILITY),
        LSTM(NOTE_MODEL_LAYER_2, return_sequences=True),
        # Dropout(DROPOUT_PROBABILITY),

        Lambda(add_dimension_3, output_shape=get_expanded_shape_3),
        Reshape((NUM_SEGMENTS * NUM_TIMESTEPS * NUM_NOTES, NOTE_MODEL_LAYER_2)),
        Lambda(remove_dimension_3, output_shape=get_contracted_shape_3),

        Dense(OUTPUT_LAYER),
        Activation('sigmoid')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    pieces = multi_training.load_pieces('music')
    piece_segment_1 = multi_training.get_piece_segment(pieces)
    piece_segment_2 = multi_training.get_piece_segment(pieces)


    # X_train = np.array([piece_segment_1[0], piece_segment_2[0]])
    X_train = np.array([np.array([piece_segment_1[0], piece_segment_2[0]])])
    print len(X_train)
    print len(X_train[0])
    print len(X_train[0][0])
    print len(X_train[0][0][0])
    print len(X_train[0][0][0][0])
    print X_train[0][0][0][0]

    y_train = np.array([np.array([piece_segment_1[1], piece_segment_2[1]])])
    print len(y_train)
    print len(y_train[0])
    print len(y_train[0][0])
    print len(y_train[0][0][0])
    print len(y_train[0][0][0][0])

    # print len(X_train)
    # print len(y_train)

    model.fit(X_train, y_train, nb_epoch=NUM_EPOCHS, batch_size=NUM_SEGMENTS)

if __name__ == '__main__':
    main()
