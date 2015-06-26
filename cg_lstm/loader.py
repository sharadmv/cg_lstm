import numpy as np

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
        for i, char in enumerate(text):
            X[i, :] = self.vocab_map[char]
        return X

    def translate(self, indices):
        return ''.join([self.reverse_vocab_map[c] for c in indices])

    def vocab_size(self):
        return len(self.vocab_map)
