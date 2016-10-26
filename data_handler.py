import numpy as np
from keras.utils import np_utils

def get_data(file, class_count):
    data = np.loadtxt(file, delimiter=',')

    examples = data[:, : - 1]

    labels = data[:, data.shape[1] - 1]
    integer_labels = [int(label) for label in labels]
    categorical_labels = np_utils.to_categorical(integer_labels, class_count)

    return examples, categorical_labels

def split_data(data, count):
    return data[0:count], data[count:data.shape[0]]
