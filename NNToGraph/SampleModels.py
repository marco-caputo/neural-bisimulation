import torch
import torch.nn as nn
import tensorflow as tf
from typing import List

class TorchFFNN(torch.nn.Module):
    def __init__(self, layers_dim: List[int]):
        """
        Initializes a simple feedforward neural network in PyTorch with ReLU activation functions.

        :param layers_dim: List of integers representing the number of neurons in each layer.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_dim) - 1):
            self.layers.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))

    def forward(self, x):
        for layer in self.layers: x = torch.relu(layer(x))
        return x

class TensorFlowFFNN(tf.keras.Model):
    def __init__(self, layers_dim: List[int]):
        """
        Initializes a simple feedforward neural network in TensorFlow with ReLU activation functions.

        :param layers_dim: List of integers representing the number of neurons in each layer.
        """
        super(TensorFlowFFNN, self).__init__()
        self.layers_list = []
        for i in range(1, len(layers_dim)):
            self.layers_list.append(tf.keras.layers.Dense(layers_dim[i], activation='relu'))

        # Pass dummy data to ensure the model is built
        dummy_input = tf.random.uniform((1, layers_dim[0]))
        _ = self(dummy_input)  # Forward pass to build the model

    def call(self, inputs, training=None):
        """
        Forward pass through the network.

        :param inputs: Input tensor.
        :param training: Boolean to specify if the model is in training mode.
        :return: Output tensor.
        """
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x