#!/usr/bin/env python
import argparse
import numpy as np
import cPickle as pickle
import repository_handler
import piece_handler
from music_generator import MusicGenerator

TIME_MODEL_LAYERS = [300, 300]
NOTE_MODEL_LAYERS = [100, 50]

SUMMARY_THRESHOLD = 5
CHECKPOINT_THRESHOLD = 5

DEFAULT_EPOCHS = 200 
DEFAULT_BATCH_SIZE = 5

CONFIGURATIONS_DIRECTORY = 'configurations/'
GENERATED_PIECES_DIRECTORY = 'generated_pieces/'

CONFIGURATION_FILE_NAME = 'configuration-%s.config'
GENERATED_PIECE_NAME = 'generated-piece-%s.mid'
LOSS_HISTORY_FILE_NAME = 'loss-history.txt'

FINAL_TAG = 'final'


def display_summary(epoch, batch_size, loss):
    print 'Epoch %d' % epoch
    print '    Loss = %d' % loss
    print '    Pieces = %d' % (epoch * batch_size)


def save_configuration(deep_jammer, tag):
    configuration_path = CONFIGURATIONS_DIRECTORY + CONFIGURATION_FILE_NAME % tag
    configuration_file = open(configuration_path, 'wb')
    pickle.dump(deep_jammer.configuration, configuration_file)


def save_loss_history(loss_history):
    f = open(LOSS_HISTORY_FILE_NAME, 'w')
    for loss in loss_history:
        f.write('%s\n' % loss)


def save_generated_piece(generated_piece, tag):
    generated_piece_path = GENERATED_PIECES_DIRECTORY + GENERATED_PIECE_NAME % tag
    piece_handler.save_piece(generated_piece, generated_piece_path)


def generate_piece(deep_jammer, pieces):
    training_example, label = map(np.array, piece_handler.get_segment(pieces))

    initial_note = training_example[0]
    generated_piece = deep_jammer.predict(piece_handler.SEGMENT_LENGTH, initial_note)

    initial_prediction = np.expand_dims(label[0], 0)
    return np.concatenate((initial_prediction, generated_piece), axis=0)


def train(deep_jammer, pieces, epochs, batch_size):
    loss_history = []

    for epoch in xrange(epochs):
        loss = deep_jammer.update(*piece_handler.get_piece_batch(pieces, batch_size))

        loss_history.append(loss)

        if epoch % SUMMARY_THRESHOLD == 0:
            display_summary(epoch, batch_size, loss)

        if epoch % CHECKPOINT_THRESHOLD == 0:
            print 'Epoch %d -> Checkpoint' % epoch
            save_configuration(deep_jammer, epoch)
            save_loss_history(loss_history)

            piece = generate_piece(deep_jammer, pieces)
            save_generated_piece(piece, epoch)


def main():
    print 'Retrieving the repository...'
    pieces = repository_handler.load_repository(args.repository)

    print 'Generating Deep Jammer...'
    deep_jammer = MusicGenerator(TIME_MODEL_LAYERS, NOTE_MODEL_LAYERS)

    print 'Training Deep Jammer...'
    train(deep_jammer, pieces, args.epochs, args.batch_size)

    print 'Saving Deep Jammer...'
    save_configuration(deep_jammer, FINAL_TAG)

    print 'Deep Jamming...'
    generated_piece = generate_piece(deep_jammer, pieces)
    save_generated_piece(generated_piece, FINAL_TAG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Builds a creative, ingenious, classical music generator.')
    parser.add_argument('repository', metavar='repository', help='the name of the repository')
    parser.add_argument('--epochs', default=DEFAULT_EPOCHS, type=int, metavar='epochs', help='the number of epochs')
    parser.add_argument('--batch_size', default=DEFAULT_BATCH_SIZE, type=int, metavar='batch_size', help='the size of each batch')

    args = parser.parse_args()

    main()
