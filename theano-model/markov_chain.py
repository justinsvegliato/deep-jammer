import argparse
import numpy as np
import piece_handler
import repository_handler


def get_probability_matrix(data):
    num_batch, num_timesteps, num_notes, num_dimensions = data.shape
    unpacked_data = data.reshape((num_batch, num_timesteps, num_notes * num_dimensions))

    matrix = np.zeros((num_notes * 2, num_notes * 2))
    normalizer = np.zeros((num_notes * 2, num_notes * 2))

    for i in xrange(num_batch):
        for j in xrange(num_timesteps - 1):
            timestep = j + 1

            previous_notes = unpacked_data[i, timestep - 1, :]
            notes = unpacked_data[i, timestep, :]

            for k in xrange(num_notes * num_dimensions):
                if previous_notes[k] == 1:
                    normalizer[k, :] += 1
                    matrix[k, :] += notes
                    
    return matrix / (normalizer + (normalizer == 0))


def generate_music(start_state, probability_matrix, segment_length):
    num_notes, num_dimensions = start_state.shape

    segment = np.zeros((segment_length, num_notes * 2))
    segment[0] = start_state.reshape((num_notes * 2))

    for i in xrange(segment_length - 1):
        mask = np.random.rand((num_notes * 2))
        next_state_probabilities = probability_matrix.dot(segment[i])

        segment[i + 1] = next_state_probabilities > mask

        segment[i + 1, :num_notes] *= segment[i + 1, num_notes:]

    return segment.reshape((segment_length, num_notes, num_dimensions))


def main():
    print 'Retrieving the repository...'
    pieces = repository_handler.load_repository(args.repository)
    _, data = piece_handler.get_piece_batch(pieces, args.batch_size)

    print 'Generating Probability Matrix..'
    probability_matrix = get_probability_matrix(data)

    print 'Making Sweet Jams...'
    segment = generate_music(data[0, 0], probability_matrix, args.segment_length)

    print 'Saving Sweet Jams...'
    piece_handler.save_piece(segment, args.file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a shitty, shitty classical music generator.')
    parser.add_argument('repository', metavar='repository', help='the name of the repository')
    parser.add_argument('file_path', metavar='filePath', help='the generated music destination')
    parser.add_argument('--batch_size', default=2000, type=int, metavar='batchSize', help='the size of each batch')
    parser.add_argument('--segment_length', default=128, type=int, metavar='segmentLength', help='the length of the generated music segment')

    args = parser.parse_args()

    main()
