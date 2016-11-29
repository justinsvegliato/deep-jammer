import os, random
import midi_parser
import data

batch_length = 128
division_length = 16

def load_pieces(path):
    pieces = {}

    for file in os.listdir(path):
        file_name = file[:-4]
        extension = file[-4:]

        if extension.lower() != '.mid':
            continue

        file_path = os.path.join(path, file)
        note_state_matrix = midi_parser.midiToNoteStateMatrix(file_path)

        if len(note_state_matrix) < batch_length:
            continue

        pieces[file_name] = note_state_matrix

    return pieces

def get_piece_segment(pieces):
    selected_pieces = random.choice(pieces.values())

    start_index = random.randrange(0, len(selected_pieces) - batch_length, division_length)
    end_index = start_index + batch_length

    output = selected_pieces[start_index:end_index]
    input = data.noteStateMatrixToInputForm(output)

    return input, output

pieces = load_pieces('music')
piece_segment = get_piece_segment(pieces)

# Playground

# This line has a length of 128. This represents each time step. In a word,
# we have 128 time steps.
# print len(piece_segment[0])

# This line has a length of 78. So, at each time step, we have 78 notes that
# can be played.
# print len(piece_segment[0][0])

# This line has a length of 78. This data structure is the output probabilities.
# That is, each tuple is of the shape (play probability, articulate probability).
# print len(piece_segment[1][0])

# In summary, for each time step (for a total of 128 time steps, we feed the
# statistics for each possible note into the neural network. We then compare
# our predicted output with the correct output and then backpropagate.