import numpy as np
import unittest
from cg_lstm import LSTM

def logistic(x):
    return 1.0/(1.0 + np.exp(-x))


class TestSimpleLSTM(unittest.TestCase):

    def setUp(self):
        self.lstm = LSTM(1, 1, num_layers=1, use_forget_gate=False).compile()
        self.lstm_forget = LSTM(1, 1, num_layers=1, use_forget_gate=True).compile()

    def zero_lstm(self, lstm):
        for param in lstm.parameters():
            param.set_value(param.get_value() * 0)

    def set_weights(self, lstm, val):
        self.zero_lstm(lstm)
        for param in lstm.parameters():
            if param.name[0] != "b":
                param.set_value(param.get_value() + val)


    def test_input_gate(self):
        self.set_weights(self.lstm, 1)
        X = np.ones((1, 1))
        S = np.zeros((1, 1, 1))
        H = np.zeros((1, 1, 1))

        input_gate = logistic(X)
        state = np.tanh(1) * input_gate
        output_gate = logistic(state + 1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = self.lstm.step(X, S, H)

        self.assertEqual(lstm_out, out)

        self.lstm.Wi.set_value(self.lstm.Wi.get_value() * 0)

        input_gate = 0.5
        state = np.tanh(1) * input_gate
        output_gate = logistic(state + 1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = self.lstm.step(X, S, H)
        self.assertEqual(lstm_out, out)


    def test_output_gate(self):
        self.set_weights(self.lstm, 1)
        X = np.ones((1, 1))
        S = np.zeros((1, 1, 1))
        H = np.zeros((1, 1, 1))

        input_gate = logistic(X)
        state = np.tanh(1) * input_gate
        output_gate = logistic(state + 1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = self.lstm.step(X, S, H)

        self.assertEqual(lstm_out, out)
        self.assertEqual(lstm_state, state)

        self.lstm.Wo.set_value(self.lstm.Wo.get_value() * 0)

        input_gate = logistic(X)
        state = np.tanh(1) * input_gate
        output_gate = logistic(state)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = self.lstm.step(X, S, H)
        self.assertEqual(lstm_out, out)
        self.assertEqual(lstm_state, state)

    def test_forget_gate(self):
        self.set_weights(self.lstm_forget, 1)
        X = np.ones((1, 1))
        S = np.ones((1, 1, 1))
        H = np.zeros((1, 1, 1))

        input_gate = logistic(X)
        forget_gate = logistic(X)
        state = np.tanh(1) * input_gate + forget_gate
        output_gate = logistic(state + 1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = self.lstm_forget.step(X, H, S)

        self.assertEqual(lstm_out, out)

        self.lstm_forget.Wf.set_value(self.lstm_forget.Wf.get_value() * 0)

        input_gate = logistic(X)
        forget_gate = 0.5
        state = np.tanh(1) * input_gate + forget_gate
        output_gate = logistic(state + 1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = self.lstm_forget.step(X, H, S)

        self.assertEqual(lstm_out, out)



if __name__ == "__main__":
    unittest.main()
