#!/usr/bin/env python
import json
import argparse
import piece_handler

REPOSITORY_DIRECTORY = 'repositories'
PIECE_DIRECTORY = 'pieces'


def save_repository(pieces, name):
    filename = '%s/%s.json' % (REPOSITORY_DIRECTORY, name)
    with open(filename, 'w') as f:
        json.dump(pieces, f)


def load_repository(name):
    filename = '%s/%s.json' % (REPOSITORY_DIRECTORY, name)
    with open(filename, 'r') as f:
        return json.load(f)


def main():
    print 'Loading the MIDI files...'
    pieces = piece_handler.get_pieces(PIECE_DIRECTORY)

    print 'Saving the repository...'
    save_repository(pieces, args.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a repository of MIDI files.')
    parser.add_argument('name', metavar='name', help='the name of the new repository')

    args = parser.parse_args()

    main()
