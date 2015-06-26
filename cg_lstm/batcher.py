import numpy as np

class Batcher(object):

    def __init__(self, X, vocab_size, sequence_length=50, batch_size=50, validation_size=0.05):
        self.X = X
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.validation_size = validation_size

        self.batch_index = 0
        N, D = self.X.shape
        assert N > self.batch_size * self.sequence_length, "File has to be at least %u characters" % (self.batch_size * self.sequence_length)

        self.X = self.X[:N - N % (self.batch_size * self.sequence_length)]
        self.N, self.D = self.X.shape
        self.X = self.X.reshape((self.N / self.sequence_length, self.sequence_length, self.D))

        self.N, self.S, self.D = self.X.shape

        idx = np.arange(self.N)
        valid_size = int(self.validation_size * self.N)

        self.X, self.Xvalid = self.X[idx[valid_size:]], self.X[idx[:valid_size]]
        self.N, self.S, self.D = self.X.shape

        self.num_sequences = self.N / self.sequence_length
        self.num_batches = self.N / self.batch_size
        self.batch_cache = {}

    def get_validation(self):
        X = self.Xvalid
        N, S, D = self.Xvalid.shape
        y = np.zeros((N, S, self.vocab_size))
        for i in xrange(N):
            for c in xrange(S):
                y[i, c, int(X[i, c, 0])] = 1

        assert (y.argmax(axis=2) == X.ravel().reshape(X.shape[:2])).all()
        y = y[:, 1:, :]
        X = X[:, :-1, :]
        X = np.swapaxes(X, 0, 1)
        y = np.swapaxes(y, 0, 1)
        return X, y

    def next_batch(self):
        idx = (self.batch_index * self.batch_size)
        if self.batch_index >= self.num_batches:
            self.batch_index = 0
            idx = 0

        if self.batch_index in self.batch_cache:
            batch = self.batch_cache[self.batch_index]
            self.batch_index += 1
            return batch

        X = self.X[idx:idx + self.batch_size]
        y = np.zeros((X.shape[0], self.sequence_length, self.vocab_size))
        for i in xrange(self.batch_size):
            for c in xrange(self.sequence_length):
                y[i, c, int(X[i, c, 0])] = 1

        assert (y.argmax(axis=2) == X.ravel().reshape(X.shape[:2])).all()
        y = y[:, 1:, :]
        X = X[:, :-1, :]

        assert y.shape[1] == self.sequence_length - 1 and X.shape[1] == self.sequence_length - 1

        X = np.swapaxes(X, 0, 1)
        y = np.swapaxes(y, 0, 1)
        self.batch_cache[self.batch_index] = X, y
        self.batch_index += 1
        return X, y

