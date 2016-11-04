import os
import numpy as np
import hdf5_handler
from keras.utils import np_utils

def get_dataset(file, class_count):
    data = np.loadtxt(file, delimiter=',')

    examples = data[:, : - 1]

    labels = data[:, data.shape[1] - 1]
    integer_labels = [int(label) for label in labels]
    categorical_labels = np_utils.to_categorical(integer_labels, class_count)

    # print 'Unique Classes', len(np.unique(labels))

    return examples, categorical_labels

def split_dataset(data, training_example_count):
    return data[0:training_example_count], data[training_example_count:data.shape[0]]

def get_file_paths(directory):
    file_paths = []
    for path, subdirs, files in os.walk(directory):
        for file in files:
            if file[0] != '.':
                file_paths.append(os.path.join(path, file))
    return file_paths

# TODO Rewrite this method to not suck - pretty sure there should be a separation of responsibilities here
def generate_dataset(directory, size, get_example, get_label):
    dataset = []
    classes = {}

    row_count = 0
    class_count = 0

    paths = get_file_paths(directory)
    for path in paths:
        if row_count == size:
            break

        h5 = hdf5_handler.open_h5_file_read(path)
        num_songs = hdf5_handler.get_num_songs(h5)

        for song_id in range(num_songs):
            label = get_label(h5, song_id)
            if label not in classes:
                classes[label] = class_count
                class_count += 1

            row = get_example(h5, song_id).tolist()
            row.append(classes[label])

            dataset.append(row)

        row_count += 1

    return dataset, classes

# TODO Remove this in favor of np.savetxt
def save_dataset(dataset):
    file = open('data.csv', 'w')
    for row in dataset:
        modified_row = [str(cell) for cell in row]
        file.write(','.join(modified_row) + '\n')
