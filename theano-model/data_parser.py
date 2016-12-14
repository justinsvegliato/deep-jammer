import itertools
import piece_handler

LOWER_VICINITY = -12
UPPER_VICINITY = 13


def get(list, index, default):
    try:
        return list[index]
    except IndexError:
        return default


def get_context(state):
    context = [0] * 12

    for note, feature in enumerate(state):
        if feature[0] == 1:
            pitch = (note + piece_handler.LOWER_BOUND) % 12
            context[pitch] += 1

    return context


def get_beat(time):
    return [2 * x - 1 for x in [time % 2, (time // 2) % 2, (time // 4) % 2, (time // 8) % 2]]


def get_input_form(note, state, context, beat):
    position_component = [note]

    pitch = (note + piece_handler.LOWER_BOUND) % 12
    pitch_component = [int(i == pitch) for i in range(12)]

    vicinity = range(LOWER_VICINITY, UPPER_VICINITY)
    previous_vicinity_component = list(itertools.chain.from_iterable((get(state, note + offset, [0, 0]) for offset in vicinity)))

    context_component = context[pitch:] + context[:pitch]

    return position_component + pitch_component + previous_vicinity_component + context_component + beat + [0]


def get_single_input_form(state, time):
    return [get_input_form(note, state, get_context(state), get_beat(time)) for note in range(len(state))]


def get_multiple_input_forms(state_matrix):
    return [get_single_input_form(state, time) for time, state in enumerate(state_matrix)]
