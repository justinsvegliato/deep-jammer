import numpy as np
import random
import os
import data_parser
import midi

LOWER_BOUND = 24
UPPER_BOUND = 102

SEGMENT_LENGTH = 128
DIVISION_LENGTH = 16

DIRECTORY = 'art'


def get_piece_batch(pieces, batch_size):
    inputs, outputs = zip(*[get_segment(pieces) for _ in range(batch_size)])
    return np.array(inputs)[:, :-1], np.array(outputs)[:, 1:]


def get_segment(pieces):
    selected_piece = random.choice(pieces.values())

    start_index = random.randrange(0, len(selected_piece) - SEGMENT_LENGTH, DIVISION_LENGTH)
    end_index = start_index + SEGMENT_LENGTH

    output = selected_piece[start_index:end_index]
    input = data_parser.get_multiple_input_forms(output)

    return input, output


def get_pieces(path, size):
    files = os.listdir(path)

    if size > len(files):
        return False

    pieces = {}
    for i in xrange(size):
        file = files[i]

        file_name = file[:-4]
        extension = file[-4:]

        if extension.lower() != '.mid':
            continue

        file_path = os.path.join(path, file)
        note_state_matrix = get_piece(file_path)

        # TODO This skips too many files that don't meet the proper shape
        if len(note_state_matrix) < SEGMENT_LENGTH:
            continue

        pieces[file_name] = note_state_matrix

    return pieces

# TODO Refactor this parsing code from the blog post
def get_piece(midi_file):
    pattern = midi.read_midifile(midi_file)

    timeleft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]

    statematrix = []
    span = UPPER_BOUND - LOWER_BOUND
    time = 0

    state = [[0, 0] for x in range(span)]
    statematrix.append(state)
    while True:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0], 0] for x in range(span)]
            statematrix.append(state)

        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < LOWER_BOUND) or (evt.pitch >= UPPER_BOUND):
                        pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch - LOWER_BOUND] = [0, 0]
                        else:
                            state[evt.pitch - LOWER_BOUND] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        # print "Found time signature event {}. Bailing!".format(evt)
                        return statematrix

                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    return statematrix

# TODO Refactor this parsing code from the blog post
def save_piece(piece, name):
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    span = UPPER_BOUND - LOWER_BOUND
    tickscale = 55

    lastcmdtime = 0
    prevstate = [[0, 0] for x in range(span)]
    for time, state in enumerate(piece + [prevstate[:]]):
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time - lastcmdtime) * tickscale, pitch=note + LOWER_BOUND))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time - lastcmdtime) * tickscale, velocity=40, pitch=note + LOWER_BOUND))
            lastcmdtime = time

        prevstate = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile('%s/%s.mid' % (DIRECTORY, name), pattern)
