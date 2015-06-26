import numpy as np
import theano
import theano.tensor as T

from theanify import theanify, Theanifiable

class Softmax(Theanifiable):

    def __init__(self, n_input, n_output):
        super(Softmax, self).__init__()
        self.n_input = n_input
        self.n_output = n_output

        assert self.n_output > 1, "Need more than 1 output for softmax"

        self.Ws = theano.shared(np.random.random((self.n_input, self.n_output)), name='Ws')
        self.bs = theano.shared(np.random.random(self.n_output), name='bs')

    @theanify(T.matrix('X'))
    def forward(self, X):
        return T.nnet.softmax(T.dot(X, self.Ws) + self.bs)

    def parameters(self):
        return [self.Ws, self.bs]

    def state(self):
        return [self.n_input, self.n_output, self.Ws, self.bs]

    @staticmethod
    def load(state):
        obj = Softmax(state[0], state[1])
        obj.Ws.set_value(state[2].get_value())
        obj.bs.set_value(state[3].get_value())
        return obj
