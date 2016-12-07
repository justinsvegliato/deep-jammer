import theano, theano.tensor as T
import numpy as np
from pass_through_layer import PassThroughLayer
from out_to_in_op import OutputFormToInputFormOp
from theano_lstm import LSTM, StackedCells, Layer, create_optimization_updates, MultiDropout


def has_hidden(layer):
    return hasattr(layer, 'initial_hidden_state')


def matrixify(vector, n):
    return T.repeat(T.shape_padleft(vector), n, axis=0)


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


def get_last_layer(result):
    if isinstance(result, list):
        return result[-1]
    else:
        return result


def ensure_list(result):
    if isinstance(result, list):
        return result
    else:
        return [result]

INPUT_SIZE = 80
OUTPUT_SIZE = 2


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

        self.multiplier = T.fscalar()
        self.random_number_generator = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))

        self._setup_train()
        self._setup_predict()
        self._setup_slow_walk()

    @property
    def params(self):
        return self.time_model.params + self.note_model.params

    @params.setter
    def params(self, param_list):
        time_model_size = len(self.time_model.params)
        self.time_model.params = param_list[:time_model_size]
        self.note_model.params = param_list[time_model_size:]

    @property
    def learned_config(self):
        return [self.time_model.params, self.note_model.params,
                [l.initial_hidden_state for mod in (self.time_model, self.note_model) for l in mod.layers if
                 has_hidden(l)]]

    @learned_config.setter
    def learned_config(self, learned_list):
        self.time_model.params = learned_list[0]
        self.note_model.params = learned_list[1]
        for l, val in zip((l for mod in (self.time_model, self.note_model) for l in mod.layers if has_hidden(l)), learned_list[2]):
            l.initial_hidden_state.set_value(val.get_value())

    def get_time_model_input(self, adjusted_input):
        batch_size, num_timesteps, num_notes, num_attributes = adjusted_input.shape

        tranposed_input = adjusted_input.transpose((1, 0, 2, 3))
        return tranposed_input.reshape((num_timesteps, batch_size * num_notes, num_attributes))

    def get_note_model_input(self, adjusted_input, adjusted_output, time_model_output):
        batch_size, num_timesteps, num_notes, _ = adjusted_input.shape
        num_hidden = time_model_output.shape[2]

        reshaped_time_model_output = time_model_output.reshape((num_timesteps, batch_size, num_notes, num_hidden))
        transposed_time_model_output = reshaped_time_model_output.transpose((2, 1, 0, 3))
        adjusted_time_model_output = transposed_time_model_output.reshape((num_notes, batch_size * num_timesteps, num_hidden))

        starting_notes = T.alloc(np.array(0, dtype=np.int8), 1, adjusted_time_model_output.shape[1], 2)
        correct_choices = adjusted_output[:, :, :-1, :].transpose((2, 0, 1, 3))
        reshaped_correct_choices = correct_choices.reshape((num_notes - 1, batch_size * num_timesteps, 2))
        adjusted_correct_choices = T.concatenate([starting_notes, reshaped_correct_choices], axis=0)

        return T.concatenate([adjusted_time_model_output, adjusted_correct_choices], axis=2)

    def get_dropout_masks(self, adjusted_input, layer_sizes):
        batch_size = adjusted_input.shape[1]
        return MultiDropout([(batch_size, shape) for shape in layer_sizes], self.dropout_probability)

    def get_outputs_info(self, adjusted_input, layers):
        batch_size = adjusted_input.shape[1]
        return [initial_state_with_taps(layer, batch_size) for layer in layers]

    def get_output(self, step, input, masks, outputs_info):
        result, _ = theano.scan(fn=step, sequences=[input], non_sequences=masks, outputs_info=outputs_info)
        return get_last_layer(result)

    def get_prediction(self, adjusted_input, note_model_output):
        batch_size, num_timesteps, num_notes, _ = adjusted_input.shape

        reshaped_note_model_output = note_model_output.reshape((num_notes, batch_size, num_timesteps, OUTPUT_SIZE))
        return reshaped_note_model_output.transpose(1, 2, 0, 3)

    def get_loss(self, adjusted_output, prediction):
        epsilon = np.spacing(np.float32(1.0))

        active_notes = T.shape_padright(adjusted_output[:, :, :, 0])
        masks = T.concatenate([T.ones_like(active_notes), active_notes], axis=3)

        log_likelihoods = T.log(2 * prediction * adjusted_output - prediction - adjusted_output + 1 + epsilon)
        masked_log_likelihoods = masks * log_likelihoods

        return T.neg(T.sum(masked_log_likelihoods))

    def _setup_train(self):
        input = T.btensor4()
        adjusted_input = input[:, :-1]

        output = T.btensor4()
        adjusted_output = output[:, 1:]

        def step_time(in_data, *other):
            other = list(other)

            split = -len(self.time_model_layer_sizes)
            hiddens = other[:split]
            masks = [None] + other[split:]

            return self.time_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)

        time_model_input = self.get_time_model_input(adjusted_input)
        time_model_masks = self.get_dropout_masks(time_model_input, self.time_model_layer_sizes)
        time_model_outputs_info = self.get_outputs_info(time_model_input, self.time_model.layers)
        time_model_output = self.get_output(step_time, time_model_input, time_model_masks, time_model_outputs_info)

        def step_note(in_data, *other):
            other = list(other)

            split = -len(self.note_model_layer_sizes)
            hiddens = other[:split]
            masks = [None] + other[split:]

            return self.note_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)

        note_model_input = self.get_note_model_input(adjusted_input, adjusted_output, time_model_output)
        note_model_masks = self.get_dropout_masks(note_model_input, self.note_model_layer_sizes)
        note_outputs_info = self.get_outputs_info(note_model_input, self.note_model.layers)
        note_model_output = self.get_output(step_note, note_model_input, note_model_masks, note_outputs_info)

        prediction = self.get_prediction(adjusted_input, note_model_output)

        loss = self.get_loss(adjusted_output, prediction)

        updates, _, _, _, _ = create_optimization_updates(loss, self.params)
        self.update_fun = theano.function(inputs=[input, output], outputs=loss, updates=updates, allow_input_downcast=True)

    def _predict_step_note(self, in_data_from_time, *states):
        hiddens = list(states[:-1])
        in_data_from_prev = states[-1]
        in_data = T.concatenate([in_data_from_time, in_data_from_prev])

        masks = [1 - self.dropout_probability for _ in self.note_model.layers]
        masks[0] = None

        new_states = self.note_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)

        probabilities = get_last_layer(new_states)

        is_note_played = self.random_number_generator.uniform() < (probabilities[0] ** self.multiplier)
        is_note_articulated = is_note_played * (self.random_number_generator.uniform() < probabilities[1])

        chosen = T.cast(T.stack(is_note_played, is_note_articulated), 'int8')

        return ensure_list(new_states) + [chosen]

    def _setup_predict(self):
        self.predict_seed = T.bmatrix()
        self.steps_to_simulate = T.iscalar()

        def step_time(*states):
            hiddens = list(states[:-2])
            in_data = states[-2]
            time = states[-1]

            masks = [1 - self.dropout_probability for _ in self.time_model.layers]
            masks[0] = None

            new_states = self.time_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)

            time_final = get_last_layer(new_states)

            start_note_values = theano.tensor.alloc(np.array(0, dtype=np.int8), 2)

            note_outputs_info = ([initial_state_with_taps(layer) for layer in self.note_model.layers] +
                                 [dict(initial=start_note_values, taps=[-1])])

            notes_result, updates = theano.scan(fn=self._predict_step_note, sequences=[time_final],
                                                outputs_info=note_outputs_info)

            output = get_last_layer(notes_result)

            next_input = OutputFormToInputFormOp()(output, time + 1)

            return (ensure_list(new_states) + [next_input, time + 1, output]), updates

        num_notes = self.predict_seed.shape[0]

        time_outputs_info = ([initial_state_with_taps(layer, num_notes) for layer in self.time_model.layers] +
                             [dict(initial=self.predict_seed, taps=[-1]),
                              dict(initial=0, taps=[-1]),
                              None])

        time_result, updates = theano.scan(fn=step_time,
                                           outputs_info=time_outputs_info,
                                           n_steps=self.steps_to_simulate)


        self.predicted_output = time_result[-1]

        self.predict_fun = theano.function(
            inputs=[self.steps_to_simulate, self.multiplier, self.predict_seed],
            outputs=self.predicted_output,
            updates=updates,
            allow_input_downcast=True)

    def _setup_slow_walk(self):
        self.walk_input = theano.shared(np.ones((2, 2), dtype='int8'))
        self.walk_time = theano.shared(np.array(0, dtype='int64'))
        self.walk_hiddens = [theano.shared(np.ones((2, 2), dtype=theano.config.floatX)) for layer in
                             self.time_model.layers if has_hidden(layer)]

        masks = [1 - self.dropout_probability for layer in self.time_model.layers]
        masks[0] = None

        new_states = self.time_model.forward(self.walk_input, prev_hiddens=self.walk_hiddens, dropout=masks)

        time_final = get_last_layer(new_states)

        start_note_values = theano.tensor.alloc(np.array(0, dtype=np.int8), 2)
        note_outputs_info = ([initial_state_with_taps(layer) for layer in self.note_model.layers] +
                             [dict(initial=start_note_values, taps=[-1])])

        notes_result, updates = theano.scan(fn=self._predict_step_note, sequences=[time_final],
                                            outputs_info=note_outputs_info)

        output = get_last_layer(notes_result)

        next_input = OutputFormToInputFormOp()(output, self.walk_time + 1)

        slow_walk_results = (new_states[:-1] + notes_result[:-1] + [next_input, output])

        updates.update({
            self.walk_time: self.walk_time + 1,
            self.walk_input: next_input
        })

        updates.update(
            {hidden: newstate for hidden, newstate, layer in zip(self.walk_hiddens, new_states, self.time_model.layers)
             if has_hidden(layer)})

        self.slow_walk_fun = theano.function(
            inputs=[self.multiplier],
            outputs=slow_walk_results,
            updates=updates,
            allow_input_downcast=True)

    def start_slow_walk(self, seed):
        seed = np.array(seed)
        num_notes = seed.shape[0]

        self.walk_time.set_value(0)
        self.walk_input.set_value(seed)
        for layer, hidden in zip((l for l in self.time_model.layers if has_hidden(l)), self.walk_hiddens):
            hidden.set_value(np.repeat(np.reshape(layer.initial_hidden_state.get_value(), (1, -1)), num_notes, axis=0))
