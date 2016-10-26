import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
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

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(training_examples, training_labels)

    print '\nStatus: %s' % model.evaluate(test_examples, test_labels)

if __name__ == '__main__':
    main()
