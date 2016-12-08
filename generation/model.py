import theano
import theano.tensor as T
import numpy as np
from pass_through_layer import PassThroughLayer
from output_transformer import OutputTransformer
from theano_lstm import StackedCells, LSTM, Layer, MultiDropout, create_optimization_updates


def matrixify(vector, n):
    return T.repeat(T.shape_padleft(vector), n, axis=0)


def has_hidden(layer):
    return hasattr(layer, INITIAL_HIDDEN_STATE_KEY)


def initial_state(layer, dimensions=None):
    if dimensions is None:
        return layer.initial_hidden_state if has_hidden(layer) else None
    else:
        return matrixify(layer.initial_hidden_state, dimensions) if has_hidden(layer) else None


def initial_state_with_taps(layer, dimensions=None):
    state = initial_state(layer, dimensions)
    if state is not None:
        return dict(initial=state, taps=[-1])
    else:
        return None


def get_list(result):
    return result if isinstance(result, list) else [result]


INPUT_SIZE = 80
OUTPUT_SIZE = 2

INITIAL_HIDDEN_STATE_KEY = 'initial_hidden_state'


class Model(object):
    def __init__(self, time_model_layer_sizes, note_model_layer_sizes, dropout_probability):
        self.time_model = StackedCells(INPUT_SIZE, celltype=LSTM, layers=time_model_layer_sizes)
        self.time_model.layers.append(PassThroughLayer())

        note_model_input_size = time_model_layer_sizes[-1] + OUTPUT_SIZE
        self.note_model = StackedCells(note_model_input_size, celltype=LSTM, layers=note_model_layer_sizes)
        self.note_model.layers.append(Layer(note_model_layer_sizes[-1], OUTPUT_SIZE, activation=T.nnet.sigmoid))

        self.time_model_layer_sizes = time_model_layer_sizes
        self.note_model_layer_sizes = note_model_layer_sizes
        self.dropout_probability = dropout_probability

        self._initialize_update_function()
        self._initialize_predict_function()

    @property
    def params(self):
        return self.time_model.params + self.note_model.params

    @params.setter
    def params(self, param_list):
        time_model_size = len(self.time_model.params)
        self.time_model.params = param_list[:time_model_size]
        self.note_model.params = param_list[time_model_size:]

    @property
    def configuration(self):
        models = [self.time_model, self.note_model]

        initial_hidden_states = []
        for model in models:
            for layer in model.layers:
                if hasattr(layer, INITIAL_HIDDEN_STATE_KEY):
                    initial_hidden_states.append(layer.initial_hidden_state)

        return [self.time_model.params, self.note_model.params, initial_hidden_states]

    @configuration.setter
    def configuration(self, configuration):
        self.time_model.params = configuration[0]
        self.note_model.params = configuration[1]

        hidden_state_layers = []
        models = [self.time_model, self.note_model]

        for model in models:
            for layer in model.layers:
                if hasattr(layer, INITIAL_HIDDEN_STATE_KEY):
                    hidden_state_layers.append(layer)

        initial_hidden_states = configuration[2]
        for layer_id in xrange(len(hidden_state_layers)):
            layer = hidden_state_layers[layer_id]
            state = initial_hidden_states[layer_id]
            layer.initial_hidden_state.set_value(state.get_value())

    @staticmethod
    def get_time_model_input(adjusted_input):
        batch_size, num_timesteps, num_notes, num_attributes = adjusted_input.shape

        tranposed_input = adjusted_input.transpose((1, 0, 2, 3))
        return tranposed_input.reshape((num_timesteps, batch_size * num_notes, num_attributes))

    @staticmethod
    def get_note_model_input(adjusted_input, adjusted_output, time_model_output):
        batch_size, num_timesteps, num_notes, _ = adjusted_input.shape
        num_hidden = time_model_output.shape[2]

        reshaped_time_model_output = time_model_output.reshape((num_timesteps, batch_size, num_notes, num_hidden))
        transposed_time_model_output = reshaped_time_model_output.transpose((2, 1, 0, 3))
        adjusted_time_model_output = transposed_time_model_output.reshape((num_notes, batch_size * num_timesteps, num_hidden))

        starting_notes = T.alloc(np.array(0, dtype=np.int8), 1, adjusted_time_model_output.shape[1], OUTPUT_SIZE)

        correct_choices = adjusted_output[:, :, :-1, :].transpose((2, 0, 1, 3))
        reshaped_correct_choices = correct_choices.reshape((num_notes - 1, batch_size * num_timesteps, OUTPUT_SIZE))
        adjusted_correct_choices = T.concatenate([starting_notes, reshaped_correct_choices], axis=0)

        return T.concatenate([adjusted_time_model_output, adjusted_correct_choices], axis=2)

    @staticmethod
    def get_outputs_info(adjusted_input, layers):
        batch_size = adjusted_input.shape[1]
        return [initial_state_with_taps(layer, batch_size) for layer in layers]

    @staticmethod
    def get_output(step, input, masks, outputs_info):
        result, _ = theano.scan(fn=step, sequences=[input], non_sequences=masks, outputs_info=outputs_info)
        return result[-1]

    @staticmethod
    def get_prediction(adjusted_input, note_model_output):
        batch_size, num_timesteps, num_notes, _ = adjusted_input.shape

        reshaped_note_model_output = note_model_output.reshape((num_notes, batch_size, num_timesteps, OUTPUT_SIZE))
        return reshaped_note_model_output.transpose(1, 2, 0, 3)

    @staticmethod
    def get_loss(adjusted_output, prediction):
        epsilon = np.spacing(np.float32(1.0))

        active_notes = T.shape_padright(adjusted_output[:, :, :, 0])
        masks = T.concatenate([T.ones_like(active_notes), active_notes], axis=3)

        log_likelihoods = T.log(2 * prediction * adjusted_output - prediction - adjusted_output + 1 + epsilon)
        masked_log_likelihoods = masks * log_likelihoods

        return T.neg(T.sum(masked_log_likelihoods))

    def get_dropout_masks(self, adjusted_input, layer_sizes):
        batch_size = adjusted_input.shape[1]
        return MultiDropout([(batch_size, shape) for shape in layer_sizes], self.dropout_probability)

    def get_prediction_drop_masks(self, layers):
        masks = [1 - self.dropout_probability for _ in layers]
        masks[0] = None
        return masks

    def _initialize_update_function(self):
        # TODO Rewrite this function
        def time_step(input, *other):
            other = list(other)

            split = -len(self.time_model_layer_sizes)
            previous_hidden_state = other[:split]
            masks = [None] + other[split:]

            return self.time_model.forward(input, prev_hiddens=previous_hidden_state, dropout=masks)

        # TODO Rewrite this function
        def note_step(input, *other):
            other = list(other)

            split = -len(self.note_model_layer_sizes)
            previous_hidden_state = other[:split]
            masks = [None] + other[split:]

            return self.note_model.forward(input, prev_hiddens=previous_hidden_state, dropout=masks)

        input = T.btensor4()
        adjusted_input = input[:, :-1]

        output = T.btensor4()
        adjusted_output = output[:, 1:]

        time_model_input = self.get_time_model_input(adjusted_input)
        time_model_masks = self.get_dropout_masks(time_model_input, self.time_model_layer_sizes)
        time_model_outputs_info = self.get_outputs_info(time_model_input, self.time_model.layers)
        time_model_output = self.get_output(time_step, time_model_input, time_model_masks, time_model_outputs_info)

        note_model_input = self.get_note_model_input(adjusted_input, adjusted_output, time_model_output)
        note_model_masks = self.get_dropout_masks(note_model_input, self.note_model_layer_sizes)
        note_outputs_info = self.get_outputs_info(note_model_input, self.note_model.layers)
        note_model_output = self.get_output(note_step, note_model_input, note_model_masks, note_outputs_info)

        prediction = self.get_prediction(adjusted_input, note_model_output)
        loss = self.get_loss(adjusted_output, prediction)

        updates, _, _, _, _ = create_optimization_updates(loss, self.params)

        self.update = theano.function(inputs=[input, output], outputs=loss, updates=updates, allow_input_downcast=True)

    def _initialize_predict_function(self):
        def predicted_note_step(time_model_output, *states):
            previous_note_model_input = states[-1]

            note_model_input = T.concatenate([time_model_output, previous_note_model_input])

            previous_hidden_state = list(states[:-1])

            masks = self.get_prediction_drop_masks(self.note_model.layers)
            note_model_output = self.note_model.forward(note_model_input, prev_hiddens=previous_hidden_state, dropout=masks)

            probabilities = note_model_output[-1]

            generator = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))
            is_note_played = probabilities[0] > generator.uniform()
            is_note_articulated = (probabilities[1] > generator.uniform()) * is_note_played

            prediction = T.cast(T.stack(is_note_played, is_note_articulated), 'int8')

            return get_list(note_model_output) + [prediction]

        def time_step(*states):
            time_model_input = states[-2]
            previous_hidden_state = list(states[:-2])
            masks = self.get_prediction_drop_masks(self.time_model.layers)
            time_model_output = self.time_model.forward(time_model_input, prev_hiddens=previous_hidden_state, dropout=masks)

            time_final = time_model_output[-1]

            initial_note = theano.tensor.alloc(np.array(0, dtype=np.int8), OUTPUT_SIZE)

            note_outputs_info = ([initial_state_with_taps(layer) for layer in self.note_model.layers] + [dict(initial=initial_note, taps=[-1])])

            notes_result, updates = theano.scan(fn=predicted_note_step, sequences=[time_final], outputs_info=note_outputs_info)

            output = notes_result[-1]
            time = states[-1]
            next_input = OutputTransformer()(output, time + 1)

            return (get_list(time_model_output) + [next_input, time + 1, output]), updates

        length = T.iscalar()
        initial_note = T.bmatrix()

        num_notes = initial_note.shape[0]

        time_outputs_info = ([initial_state_with_taps(layer, num_notes) for layer in self.time_model.layers] + [dict(initial=initial_note, taps=[-1]), dict(initial=0, taps=[-1]), None])

        time_result, updates = theano.scan(fn=time_step, outputs_info=time_outputs_info, n_steps=length)

        prediction = time_result[-1]

        self.predict = theano.function([length, initial_note], outputs=prediction, updates=updates, allow_input_downcast=True)