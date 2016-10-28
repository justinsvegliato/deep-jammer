#!/usr/bin/env python
from data import data_handler
from sklearn.neighbors import KNeighborsClassifier

CLASS_COUNT = 2
TRAINING_EXAMPlE_COUNT = 200

def main():
    examples, labels = data_handler.get_data('../data/example_data.csv', CLASS_COUNT)

    training_examples, test_examples = data_handler.split_data(examples, TRAINING_EXAMPlE_COUNT)
    training_labels, test_labels = data_handler.split_data(labels, TRAINING_EXAMPlE_COUNT)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(training_examples, training_labels)

    print 'Score %s' % classifier.score(test_examples, test_labels)

if __name__ == '__main__':
    main()
