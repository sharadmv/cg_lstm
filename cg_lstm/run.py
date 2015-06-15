from lstm import LSTM
from cg import CharacterGenerator, Softmax

import logging
logging.basicConfig()
import cPickle as pickle
import numpy as np
from argparse import ArgumentParser
from path import Path

import theano
theano.config.reoptimize_unpickled_function = False

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

class InputMapper(object):

    def __init__(self):
        self.vocab_map = {}
        self.reverse_vocab_map = {}
        self.vocab_index = 0

    def load(self, text):
        for char in text:
            if char not in self.vocab_map:
                self.vocab_map[char] = self.vocab_index
                self.reverse_vocab_map[self.vocab_index] = char
                self.vocab_index += 1

    def convert_to_tensor(self, text):
        self.load(text)
        N = len(text)
        X = np.zeros((N, 1))
        for i, char in enumerate(text): X[i, :] = self.vocab_map[char]
        return X

    def translate(self, indices):
        return ''.join([self.reverse_vocab_map[c] for c in indices])

    def vocab_size(self):
        return len(self.vocab_map)

class Batcher(object):

    def __init__(self, X, vocab_size, sequence_length=50, batch_size=50):
        self.X = X
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.batch_index = 0
        N, D = self.X.shape
        assert N > self.batch_size * self.sequence_length, "File has to be at least %u characters" % (self.batch_size * self.sequence_length)

        self.X = self.X[:N - N % (self.batch_size * self.sequence_length)]
        self.N, self.D = self.X.shape
        self.X = self.X.reshape((self.N / self.sequence_length, self.sequence_length, self.D))

        self.num_sequences = self.N / self.sequence_length
        self.num_batches = self.N / self.batch_size

        self.batch_cache = {}

    def next_batch(self):
        idx = (self.batch_index * self.batch_size)
        if idx >= self.num_batches:
            self.batch_index = 0
            idx = 0

        if self.batch_index in self.batch_cache:
            return self.batch_cache[self.batch_index]

        X = self.X[idx:idx + self.batch_size]
        y = np.zeros((self.batch_size, self.sequence_length, self.vocab_size))
        for i in xrange(self.batch_size):
            for c in xrange(self.sequence_length):
                y[i, c, int(X[i, c, 0])] = 1

        y = y[1:, :, :]
        X = X[:-1, :, :]

        self.batch_cache[self.batch_index] = X, y
        self.batch_size += 1
        return X, y

def convert_to_list(g):
    return g.ravel()

def main(args):
    pass

if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument('input')
    argparser.add_argument('--batch_size', default=50, type=int)
    argparser.add_argument('--sequence_length', default=50, type=int)
    argparser.add_argument('--hidden_layer_size', default=128, type=int)
    argparser.add_argument('--num_layers', default=2, type=int)
    argparser.add_argument('--compiled_output', default='cg.pkl')
    argparser.add_argument('--iterations', default=20, type=int)
    argparser.add_argument('--compile', action='store_true')
    argparser.add_argument('--load', type=str)

    args = argparser.parse_args()
    main(args)

    logger.info("Loading input file...")
    loader = InputMapper()
    with open(args.input) as fp:
        text = fp.read()
    X = loader.convert_to_tensor(text)
    batcher = Batcher(X, loader.vocab_size(), batch_size=args.batch_size, sequence_length=args.sequence_length)

    cache_location = args.compiled_output if not args.compile else None

    if args.load:
        logger.info("Loading LSTM model from file...")
        cg = CharacterGenerator.load(args.load).compile(cache=cache_location)
    else:
        lstm = LSTM(1, args.hidden_layer_size, num_layers=args.num_layers)
        softmax = Softmax(args.hidden_layer_size, loader.vocab_size())
        cg = CharacterGenerator(lstm, softmax).compile(cache=cache_location)

    logger.info("Running SGD")
    learning_rate = 0.1
    N = args.batch_size
    H = args.hidden_layer_size
    L = args.num_layers
    input_state = np.zeros((N, L, H))
    def iterate(num_iterations, state):
        losses = []
        for i in xrange(num_iterations):
            batch_x, batch_y = batcher.next_batch()
            loss, state = cg.rmsprop(batch_x, batch_y, state)
            losses.append(loss)
            print loss
        return state, losses
    state, l = iterate(args.iterations, input_state)
