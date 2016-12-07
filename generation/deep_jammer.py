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

INITIAL_MULTIPLIER = 1

DEFAULT_EPOCHS = 10000
DEFAULT_BATCH_SIZE = 2
DEFAULT_LENGTH = 5

ART_DIRECTORY = 'art/'
WEIGHTS_DIRECTORY = 'weights/'
SAMPLE_DIRECTORY = 'samples/'

WEIGHTS_FILE_NAME = 'weights-%s.p'
SAMPLE_FILE_NAME = 'sample-%s.mid'

SUMMARY_THRESHOLD = 1
CHECKPOINT_THRESHOLD = 100


def display_summary(epoch, batch_size, loss):
    print 'Epoch %d' % epoch
    print '    Loss = %d' % loss
    print '    Pieces = %d' % (epoch * batch_size)


def save_weights(model, epoch=False):
    tag = epoch if epoch else 'final'

    weights_path = WEIGHTS_DIRECTORY + WEIGHTS_FILE_NAME % tag
    weights_file = open(weights_path, 'wb')

    pickle.dump(model.learned_config, weights_file)


def save_sample(model, epoch, pieces):
    input, output = map(np.array, piece_handler.get_segment(pieces))

    # TODO Do we need the extra parenthesis?
    sample_art = np.concatenate((np.expand_dims(output[0], 0), model.predict_fun(piece_handler.SEGMENT_LENGTH, 1, input[0])), axis=0)
    sample_art_path = SAMPLE_DIRECTORY + SAMPLE_FILE_NAME % epoch

    piece_handler.save_piece(sample_art, sample_art_path)


def train(model, pieces, epochs, batch_size):
    for epoch in xrange(epochs):
        loss = model.update_fun(*piece_handler.get_piece_batch(pieces, batch_size))

        if epoch % SUMMARY_THRESHOLD == 0:
            display_summary(epoch, batch_size, loss)

        if epoch % CHECKPOINT_THRESHOLD == 0:
            print 'Epoch %s: Saving checkpoint...' % epoch
            save_weights(model, epoch)
            save_sample(model, epoch, pieces)


def get_updated_multiplier(multiplier, results):
    note_count = np.sum(results[-1][:, 0])

    if note_count < 2:
        if multiplier > 1:
            return 1
        return multiplier - 0.02

    adjustment = 0.3 * (1 - multiplier)
    return multiplier + adjustment


def compose(model, pieces, length, name):
    # TODO Simplify the int8 conversion
    input, output = map(lambda x: np.array(x, dtype='int8'), piece_handler.get_segment(pieces))

    model.start_slow_walk(input[0])
    art = [output[0]]
    multiplier = INITIAL_MULTIPLIER

    for time in range(length * piece_handler.SEGMENT_LENGTH):
        results = model.slow_walk_fun(multiplier)
        multiplier = get_updated_multiplier(multiplier, results)
        art.append(results[-1])

    art_path = ART_DIRECTORY + name + '.mid'
    piece_handler.save_piece(art, art_path)


def main():
    print 'Retrieving the repository...'
    pieces = repository_handler.load_repository(args.repository)

    print 'Generating Deep Jammer...'
    deep_jammer = model.Model(TIME_MODEL_LAYERS, NOTE_MODEL_LAYERS, DROPOUT_PROBABILITY)

    print 'Training Deep Jammer...'
    train(deep_jammer, pieces, args.epochs, args.batch_size)

    print 'Saving Deep Jammer...'
    save_weights(deep_jammer)

    print 'Deep jamming...'
    compose(deep_jammer, pieces, args.length, args.piece)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compose a creative, ingenious, classical piece.')
    parser.add_argument('piece', metavar='piece', help='the name of the new piece')
    parser.add_argument('repository', metavar='repository', help='the name of the repository')
    parser.add_argument('--epochs', default=DEFAULT_EPOCHS, type=int, metavar='epochs', help='the number of epochs')
    parser.add_argument('--batch_size', default=DEFAULT_BATCH_SIZE, type=int, metavar='batchSize', help='the size of each batch')
    parser.add_argument('--length', default=DEFAULT_LENGTH, type=int, metavar='length', help='the length of the new piece')

    args = parser.parse_args()

    main()
