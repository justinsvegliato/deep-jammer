import theano
import theano.tensor as T
import numpy as np
from theano_lstm import StackedCells, LSTM, Layer, create_optimization_updates
from router import Router
from output_transformer import OutputTransformer

INPUT_SIZE = 80
OUTPUT_SIZE = 2

INITIAL_HIDDEN_STATE_KEY = 'initial_hidden_state'


class MusicGenerator(object):
    def __init__(self, time_model_layer_sizes, note_model_layer_sizes):
        self.time_model = StackedCells(INPUT_SIZE, celltype=LSTM, layers=time_model_layer_sizes)
        self.time_model.layers.append(Router())

        note_model_input_size = time_model_layer_sizes[-1] + OUTPUT_SIZE
        self.note_model = StackedCells(note_model_input_size, celltype=LSTM, layers=note_model_layer_sizes)
        self.note_model.layers.append(Layer(note_model_layer_sizes[-1], OUTPUT_SIZE, activation=T.nnet.sigmoid))

        self.time_model_layer_sizes = time_model_layer_sizes
        self.note_model_layer_sizes = note_model_layer_sizes

        self._initialize_update_function()
        self._initialize_predict_function()

    @property
    def params(self):
        return self.time_model.params + self.note_model.params

    @params.setter
    def params(self, params):
        time_model_size = len(self.time_model.params)
        self.time_model.params = params[:time_model_size]
        self.note_model.params = params[time_model_size:]

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

        starting_notes = T.alloc(0, 1, adjusted_time_model_output.shape[1], OUTPUT_SIZE)
        correct_choices = adjusted_output[:, :, :-1, :].transpose((2, 0, 1, 3))
        reshaped_correct_choices = correct_choices.reshape((num_notes - 1, batch_size * num_timesteps, OUTPUT_SIZE))
        adjusted_correct_choices = T.concatenate([starting_notes, reshaped_correct_choices], axis=0)

        return T.concatenate([adjusted_time_model_output, adjusted_correct_choices], axis=2)

    @staticmethod
    def get_initial_state(layer, dimensions=None):
        if not hasattr(layer, INITIAL_HIDDEN_STATE_KEY):
            return None

        return {
            'initial': layer.initial_hidden_state if dimensions is None else T.repeat(T.shape_padleft(layer.initial_hidden_state), dimensions, axis=0),
            'taps': [-1]
        }

    @staticmethod
    def get_output(step, input, outputs_info):
        result, _ = theano.scan(fn=step, sequences=[input], outputs_info=outputs_info)
        return result[-1]

    @staticmethod
    def get_prediction(adjusted_input, note_model_output):
        batch_size, num_timesteps, num_notes, _ = adjusted_input.shape

        reshaped_note_model_output = note_model_output.reshape((num_notes, batch_size, num_timesteps, OUTPUT_SIZE))
        return reshaped_note_model_output.transpose(1, 2, 0, 3)

    @staticmethod
    def get_loss(adjusted_output, prediction):
        epsilon = 1e-7

        active_notes = T.shape_padright(adjusted_output[:, :, :, 0])
        masks = T.concatenate([T.ones_like(active_notes), active_notes], axis=3)

        log_likelihoods = T.log(2 * prediction * adjusted_output - prediction - adjusted_output + 1 + epsilon)
        masked_log_likelihoods = masks * log_likelihoods

        return T.neg(T.sum(masked_log_likelihoods))

    def get_outputs_info(self, adjusted_input, layers):
        batch_size = adjusted_input.shape[1]
        return [self.get_initial_state(layer, batch_size) for layer in layers]

    def get_time_prediction_outputs_info(self, initial_note):
        initial_states = [self.get_initial_state(layer) for layer in self.note_model.layers]
        first_note = {
            'initial': initial_note,
            'taps': [-1]
        }
        return initial_states + [first_note]

    def get_prediction_outputs_info(self, num_notes, initial_note):
        initial_states = [self.get_initial_state(layer, num_notes) for layer in self.time_model.layers]
        first_note = {
            'initial': initial_note,
            'taps': [-1]
        }
        padder = {
            'initial': 0,
            'taps': [-1]
        }
        return initial_states + [first_note, padder, None]

    def _initialize_update_function(self):
        def time_step(input, *previous_hidden_state):
            return self.time_model.forward(input, prev_hiddens=previous_hidden_state)

        def note_step(input, *previous_hidden_state):
            return self.note_model.forward(input, prev_hiddens=previous_hidden_state)

        input = T.btensor4()
        adjusted_input = input[:, :-1]

        output = T.btensor4()
        adjusted_output = output[:, 1:]

        time_model_input = self.get_time_model_input(adjusted_input)
        time_model_outputs_info = self.get_outputs_info(time_model_input, self.time_model.layers)
        time_model_output = self.get_output(time_step, time_model_input, time_model_outputs_info)

        note_model_input = self.get_note_model_input(adjusted_input, adjusted_output, time_model_output)
        note_outputs_info = self.get_outputs_info(note_model_input, self.note_model.layers)
        note_model_output = self.get_output(note_step, note_model_input, note_outputs_info)

        prediction = self.get_prediction(adjusted_input, note_model_output)
        loss = self.get_loss(adjusted_output, prediction)

        updates, _, _, _, _ = create_optimization_updates(loss, self.params)

        self.update = theano.function(inputs=[input, output], outputs=loss, updates=updates, allow_input_downcast=True)

    def _initialize_predict_function(self):
        def predicted_note_step(time_model_output, *states):
            previous_note_model_input = states[-1]

            note_model_input = T.concatenate([time_model_output, previous_note_model_input])
            previous_hidden_state = list(states[:-1])
            note_model_output = self.note_model.forward(note_model_input, prev_hiddens=previous_hidden_state)
            probabilities = note_model_output[-1]

            generator = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))

            is_note_played = probabilities[0] > generator.uniform()
            is_note_articulated = (probabilities[1] > generator.uniform()) * is_note_played
            prediction = T.cast(T.stack(is_note_played, is_note_articulated), 'int8')

            return note_model_output + [prediction]

        def predicted_time_step(*states):
            time_model_input = states[-2]
            previous_hidden_state = list(states[:-2])
            time_model_output = self.time_model.forward(time_model_input, prev_hiddens=previous_hidden_state)

            time_model_output_last_layer = time_model_output[-1]
            initial_note = T.alloc(0, OUTPUT_SIZE)
            note_outputs_info = self.get_time_prediction_outputs_info(initial_note)
            notes_model_output, updates = theano.scan(fn=predicted_note_step, sequences=[time_model_output_last_layer], outputs_info=note_outputs_info)

            output = notes_model_output[-1]
            time = states[-1]
            next_input = OutputTransformer()(output, time + 1)

            return (time_model_output + [next_input, time + 1, output]), updates

        length = T.iscalar()
        initial_note = T.bmatrix()

        num_notes = initial_note.shape[0]
        time_outputs_info = self.get_prediction_outputs_info(num_notes, initial_note)
        time_model_output, updates = theano.scan(fn=predicted_time_step, outputs_info=time_outputs_info, n_steps=length)
        prediction = time_model_output[-1]

        self.predict = theano.function([length, initial_note], outputs=prediction, updates=updates, allow_input_downcast=True)