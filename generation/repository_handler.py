#!/usr/bin/env python
import json
import argparse
import piece_handler

REPOSITORY_DIRECTORY = 'repositories'
PIECE_DIRECTORY = 'pieces'


def save_repository(pieces, name):
    filename = '%s/%s.json' % (REPOSITORY_DIRECTORY, name)
    with open(filename, 'w') as file:
        json.dump(pieces, file)


def load_repository(name):
    filename = '%s/%s.json' % (REPOSITORY_DIRECTORY, name)
    with open(filename, 'r') as file:
        return json.load(file)


def main():
    print 'Loading the MIDI files...'
    pieces = piece_handler.get_pieces(PIECE_DIRECTORY, args.size)

    print 'Saving the repository...'
    save_repository(pieces, args.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a repository of MIDI files.')
    parser.add_argument('name', metavar='name', help='the name of the new repository')
    parser.add_argument('--size', default=10, type=int, metavar='size', help='the number of MIDI files in the repository')
    args = parser.parse_args()

    main()
