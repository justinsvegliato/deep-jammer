import itertools
import piece_handler


def get_or_default(l, i, d):
    try:
        return l[i]
    except IndexError:
        return d


def get_context(state):
    context = [0] * 12
    for note, notestate in enumerate(state):
        if notestate[0] == 1:
            pitchclass = (note + piece_handler.LOWER_BOUND) % 12
            context[pitchclass] += 1
    return context


def get_beat(time):
    return [2 * x - 1 for x in [time % 2, (time // 2) % 2, (time // 4) % 2, (time // 8) % 2]]


def get_input_form(note, state, context, beat):
    position = note
    part_position = [position]

    pitchclass = (note + piece_handler.LOWER_BOUND) % 12
    part_pitchclass = [int(i == pitchclass) for i in range(12)]
    # Concatenate the note states for the previous vicinity
    part_prev_vicinity = list(
        itertools.chain.from_iterable((get_or_default(state, note + i, [0, 0]) for i in range(-12, 13))))

    part_context = context[pitchclass:] + context[:pitchclass]

    return part_position + part_pitchclass + part_prev_vicinity + part_context + beat + [0]


def get_single_input_form(state, time):
    beat = get_beat(time)
    context = get_context(state)
    return [get_input_form(note, state, context, beat) for note in range(len(state))]


def get_multiple_input_forms(statematrix):
    return [get_single_input_form(state, time) for time, state in enumerate(statematrix)]

