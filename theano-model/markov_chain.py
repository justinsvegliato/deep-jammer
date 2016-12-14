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

    # this prevents division by zero errors, but does not affect the results.
    # Anywhere that Normalizer is 0 corresponds to an element in the Matrix that is also 0.
    return matrix / (normalizer + (normalizer == 0))


def Generate_Music(start_state, Probability_Matrix, Segment_Length):
    # Assuming Start_State is of shape [Num_Notes, 2]
    Num_Notes, Num_Dimensions = start_state.shape

    Segment = np.zeros((Segment_Length, Num_Notes * 2))
    Segment[0] = start_state.reshape((Num_Notes * 2))

    for i in xrange(Segment_Length - 1):
        Mask = np.random.rand((Num_Notes * 2))
        Next_State_Probabilities = Probability_Matrix.dot(Segment[i])

        Segment[i + 1] = Next_State_Probabilities > Mask

        Segment[i + 1, :Num_Notes] *= Segment[i + 1, Num_Notes:]

    return Segment.reshape((Segment_Length, Num_Notes, Num_Dimensions))


def main():
    print 'Retrieving the repository...'
    pieces = repository_handler.load_repository(args.repository)
    _, data = piece_handler.get_piece_batch(pieces, args.batch_size)

    print 'Generating Probability Matrix..'
    Probability_Matrix = get_probability_matrix(data)

    print 'Making Sweet Jams...'
    segment = Generate_Music(data[0, 0], Probability_Matrix, args.segment_length)

    print 'Saving Sweet Jams...'
    piece_handler.save_piece(segment, args.file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a shitty, shitty classical music generator.')
    parser.add_argument('repository', metavar='repository', help='the name of the repository')
    parser.add_argument('file_path', metavar='filePath', help='the generated music destination')
    parser.add_argument('--batch_size', default=2000, type=int, metavar='batchSize', help='the size of each batch')
    parser.add_argument('--segment_length', default=128, type=int, metavar='segmentLength',
                        help='the length of the generated music segment')

    args = parser.parse_args()

    main()
