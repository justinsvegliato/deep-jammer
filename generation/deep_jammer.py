#!/usr/bin/env python
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Reshape, Permute, Lambda

NUM_EPOCHS = 10
NUM_SEGMENTS = 10
NUM_NOTES = 10
NUM_TIMESTEPS = 10
NUM_FEATURES = 10

TIME_MODEL_LAYER_1 = 300
TIME_MODEL_LAYER_2 = 300

NOTE_MODEL_LAYER_1 = 100
NOTE_MODEL_LAYER_2 = 50

OUTPUT_LAYER = 2

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
remove_dimension_3 = lambda x: x.reshape([NUM_SEGMENTS, NUM_TIMESTEPS, NUM_NOTES, NOTE_MODEL_LAYER_2])
get_contracted_shape_3 = lambda shape: [NUM_SEGMENTS, NUM_TIMESTEPS, NUM_NOTES, NOTE_MODEL_LAYER_2]

def main():
    model = Sequential([
        Lambda(add_dimension_1, output_shape=get_expanded_shape_1, input_shape=(NUM_SEGMENTS, NUM_TIMESTEPS, NUM_NOTES, NUM_FEATURES)),
        Permute((1, 3, 2, 4)),
        Reshape((NUM_SEGMENTS * NUM_NOTES, NUM_TIMESTEPS, NUM_FEATURES)),
        Lambda(remove_dimension_1, output_shape=get_contracted_shape_1),

        LSTM(TIME_MODEL_LAYER_1, return_sequences=True),
        LSTM(TIME_MODEL_LAYER_2, return_sequences=True),

        Lambda(add_dimension_2, output_shape=get_expanded_shape_2),
        Reshape((NUM_SEGMENTS, NUM_NOTES, NUM_TIMESTEPS, TIME_MODEL_LAYER_2)),
        Permute((1, 3, 2, 4)),
        Reshape((NUM_SEGMENTS * NUM_TIMESTEPS, NUM_NOTES, TIME_MODEL_LAYER_2)),
        Lambda(remove_dimension_2, output_shape=get_contracted_shape_2),

        LSTM(NOTE_MODEL_LAYER_1, return_sequences=True),
        LSTM(NOTE_MODEL_LAYER_2, return_sequences=True),

        TimeDistributed(Dense(OUTPUT_LAYER)),
        Activation('sigmoid')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # X_train = None
    # y_train = None
    # model.fit(X_train, y_train, nb_epoch=NUM_EPOCHS, batch_size=NUM_SEGMENTS)

if __name__ == '__main__':
    main()

