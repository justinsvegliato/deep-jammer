#!/usr/bin/env python
import data_handler
from keras.models import Sequential
from keras.layers.core import Dense, Activation

CLASS_COUNT = 2
TRAINING_EXAMPlE_COUNT = 200

def main():
    examples, labels = data_handler.get_data('data.csv', CLASS_COUNT)

    training_examples, test_examples = data_handler.split_data(examples, TRAINING_EXAMPlE_COUNT)
    training_labels, test_labels = data_handler.split_data(labels, TRAINING_EXAMPlE_COUNT)

    model = Sequential([
        Dense(CLASS_COUNT, input_dim=training_examples.shape[1]),
        Activation('softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(training_examples, training_labels)
    model.evaluate(test_examples, test_labels)

if __name__ == '__main__':
    main()
