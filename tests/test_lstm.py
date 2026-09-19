import numpy as np
import torch

from src.lstm import LSTMClassifier


def test_lstm_forward_returns_three_class_logits():
    model = LSTMClassifier(input_size=4, hidden_size=8)
    outputs = model(torch.as_tensor(np.zeros((2, 5, 4)), dtype=torch.float32))
    assert outputs.shape == (2, 3)