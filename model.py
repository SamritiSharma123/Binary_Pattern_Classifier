# model.py

import torch
import torch.nn as nn


class Perceptron(nn.Module):
    """
    Single Layer Perceptron for Binary Classification
    """

    def __init__(self):
        super(Perceptron, self).__init__()

        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x