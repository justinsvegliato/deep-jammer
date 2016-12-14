import theano
import theano.tensor as T
import numpy as np
import data_parser


class OutputTransformer(theano.Op):
    __props__ = ()

    def make_node(self, state, time):
        state = T.as_tensor_variable(state)
        time = T.as_tensor_variable(time)
        return theano.Apply(self, [state, time], [T.bmatrix()])

    def perform(self, node, inputs_storage, output_storage):
        state, time = inputs_storage
        output_storage[0][0] = np.array(data_parser.get_single_input_form(state, time), dtype='int8')

    def R_op(self, inputs, eval_points):
        pass
