#!/usr/bin/env python
import argparse
import numpy as np
import cPickle as pickle
import repository_handler
import piece_handler
import model

TIME_MODEL_LAYERS = [300, 300]
NOTE_MODEL_LAYERS = [100, 50]
DROPOUT_PROBABILITY = 0.5

ART_DIRECTORY = 'art'

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 10
DEFAULT_LENGTH = 5


def train(model, pieces, epochs, batch_size):
    for i in range(epochs):
        error = model.update_fun(*piece_handler.get_piece_batch(pieces, batch_size))

        if i % 1 == 0:
            print 'Epoch %s: Error = %s' % (i, error)

        if i % 100 == 0:
            input, output = map(np.array, piece_handler.get_segment(pieces))

            piece_handler.save_piece(
                np.concatenate((np.expand_dims(output[0], 0), model.predict_fun(piece_handler.SEGMENT_LENGTH, 1, input[0])), axis=0),
                'art/sample{}'.format(i))
            pickle.dump(model.learned_config, open('art/params{}.p'.format(i), 'wb'))


def compose(model, pieces, length, name):
    input, output = map(lambda x: np.array(x, dtype='int8'), piece_handler.get_segment(pieces))

    outputs = [output[0]]

    model.start_slow_walk(input[0])

    multiplier = 1
    for time in range(length * piece_handler.SEGMENT_LENGTH):
        results = model.slow_walk_fun(multiplier)

        note_count = np.sum(results[-1][:, 0])
        if note_count < 2:
            multiplier = 1 if multiplier > 1 else multiplier - 0.02
        else:
            multiplier += (1 - multiplier) * 0.3

        outputs.append(results[-1])

    piece_handler.save_piece(np.array(outputs), 'art/' + name)


def main():
    print 'Retrieving the repository...'
    pieces = repository_handler.load_repository(args.repository)

    print 'Generating the model...'
    # TODO Rename model
    m = model.Model(TIME_MODEL_LAYERS, NOTE_MODEL_LAYERS, dropout=DROPOUT_PROBABILITY)

    print 'Training the model...'
    train(m, pieces, args.epochs, args.batch_size)

    print 'Saving the model...'
    # TODO Put this in a utility file
    filename = '%s/%s-weights.p' % (ART_DIRECTORY, args.piece)
    pickle.dump(m.learned_config, open(filename, 'wb'))

    print 'Composing a piece...'
    compose(m, pieces, args.length, args.piece)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compose a creative, ingenious, classical piece.')
    parser.add_argument('piece', metavar='piece', help='the name of the new piece')
    parser.add_argument('repository', metavar='repository', help='the name of the repository')
    parser.add_argument('--epochs', default=DEFAULT_EPOCHS, type=int, metavar='epochs', help='the number of epochs')
    parser.add_argument('--batch_size', default=DEFAULT_BATCH_SIZE, type=int, metavar='batchSize', help='the size of each batch')
    parser.add_argument('--length', default=DEFAULT_LENGTH, type=int, metavar='length', help='the length of the new piece')
    args = parser.parse_args()

    main()













