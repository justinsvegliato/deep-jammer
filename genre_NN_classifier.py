import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

CLASS_COUNT = 2
TRAINING_EXAMPlE_COUNT = 200

def get_data(file):
    data = np.loadtxt(file, delimiter=',')

    examples = data[:, : - 1]
    labels = np_utils.to_categorical(data[:, data.shape[1] - 1], CLASS_COUNT)

    return examples, labels

def split_data(data, count):
    return data[0:count], data[count:data.shape[0]]

def main():
    examples, labels = get_data('data.csv')

    training_examples, test_examples = split_data(examples, TRAINING_EXAMPlE_COUNT)
    training_labels, test_labels = split_data(labels, TRAINING_EXAMPlE_COUNT)

    model = Sequential([
        Dense(CLASS_COUNT, input_dim=training_examples.shape[1]),
        Activation('softmax')
    ])
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.fit(training_examples, training_labels)

    print 'Score: %s' % model.evaluate(test_examples, test_labels)

if __name__ == '__main__':
    main()

# Scratch:
# The data has the shape N X T, where
# N is the number of training examples, and
# T is the number of seconds multiplied by the frequency in hertz.
