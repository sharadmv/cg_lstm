from lstm import LSTM
from cg import CharacterGenerator, Softmax
from batcher import Batcher
from loader import InputMapper

import logging
logging.basicConfig()
import numpy as np
from argparse import ArgumentParser

import theano
theano.config.reoptimize_unpickled_function = False

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
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

    Xvalid, yvalid = batcher.get_validation()

    cache_location = args.compiled_output if not args.compile else None

    if args.load:
        logger.info("Loading LSTM model from file...")
        cg = CharacterGenerator.load_model(args.load).compile(cache=cache_location)
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
            print "Batch #%u: %f" % (batcher.batch_index, loss)
        return state, losses
    state, l = iterate(args.iterations, input_state)
