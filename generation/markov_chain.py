import argparse
import numpy as np
import data_parser
import piece_handler
import repository_handler

def Create_Probability_Matrix(Data):
	
	# Data should be of shape [Num_Batch, Num_Timesteps, Num_Notes, 2]

	Num_Batch, Num_Timesteps, Num_Notes, Num_Dimensions = Data.shape

	Unpacked_Data = Data.reshape((Num_Batch, Num_Timesteps,Num_Notes * Num_Dimensions))

	Matrix = np.zeros((Num_Notes * 2, Num_Notes * 2))

	Normalizer = np.zeros((Num_Notes * 2,Num_Notes * 2))

	for i in xrange(Num_Batch):
		for j in xrange(Num_Timesteps - 1):
			Time = j + 1

			Notes = Unpacked_Data[i, Time, :]
			Previous_Notes = Unpacked_Data[i, Time - 1, :]

			for k in xrange(Num_Notes * Num_Dimensions):
				if Previous_Notes[k] == 1:
					
					#Dimension 0 is the previous note, dimension 1 is the current note.
					Normalizer[k,:] += 1
					Matrix[k, :] += Notes

	# this prevents division by zero errors, but does not affect the results. 
	# Anywhere that Normalizer is 0 corresponds to an element in the Matrix that is also 0.
	return Matrix / (Normalizer + (Normalizer == 0))

def Generate_Music(start_state, Probability_Matrix, Segment_Length):
	#Assuming Start_State is of shape [Num_Notes, 2]
	Num_Notes, Num_Dimensions = start_state.shape

	Segment = np.zeros((Segment_Length, Num_Notes * 2))
	Segment[0] = start_state.reshape((Num_Notes * 2))


	for i in xrange(Segment_Length - 1):
		Mask = np.random.rand((Num_Notes * 2))
		Next_State_Probabilities = Probability_Matrix.dot(Segment[i])

		Segment[i + 1] = Next_State_Probabilities > Mask  

		Segment[i + 1, :Num_Notes] *= Segment[i + 1, Num_Notes:]
	
	return Segment.reshape((Segment_Length,Num_Notes,Num_Dimensions))

def main():
    print 'Retrieving the repository...'
    pieces = repository_handler.load_repository(args.repository)
    _, data = piece_handler.get_piece_batch(pieces, args.batch_size)

    print 'Generating Probability Matrix..'
    Probability_Matrix = Create_Probability_Matrix(data)

    print 'Making Sweet Jams...'
    segment = Generate_Music(data[0,0], Probability_Matrix, args.segment_length)

    print 'Saving Sweet Jams...'
    piece_handler.save_piece(segment, args.file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a shitty, shitty classical music generator.')
    parser.add_argument('repository', metavar='repository', help='the name of the repository')
    parser.add_argument('file_path', metavar='filePath', help='the generated music destination')
    parser.add_argument('--batch_size',default=2000, type=int, metavar='batchSize', help='the size of each batch')
    parser.add_argument('--segment_length', default=128, type=int, metavar='segmentLength', help='the length of the generated music segment')
    parser

    args = parser.parse_args()

    main()
