import torch
import torch.nn as nn
import tensorflow as tf
from typing import List

class TorchFFNN(torch.nn.Module):
    """
    A simple feedforward neural network in PyTorch with ReLU activation functions.
    """

    def __init__(self, layers_dim: List[int], activations_as_layers: bool = True):
        """
        Initializes a simple feedforward neural network in PyTorch with ReLU activation functions.

        :param layers_dim: List of integers representing the number of neurons in each linear layer.
        :param activations_as_layers: specifies if activation functions should be defined as layers. Defaults to True.
        """
        super().__init__()
        self.layers_list = nn.ModuleList()
        self.activations_as_layers = activations_as_layers
        for i in range(len(layers_dim) - 1):
            if self.activations_as_layers:
                self.layers_list.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
                if i < len(layers_dim) - 2:
                    self.layers_list.append(nn.ReLU())
            else:
                self.layers_list.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))

    def forward(self, x):
        for layer in self.layers_list: x = layer(x) if self.activations_as_layers else torch.relu(layer(x))
        return x

class TensorFlowFFNN(tf.keras.Model):
    """
    A simple feedforward neural network in TensorFlow with ReLU activation functions
    """

    def __init__(self, layers_dim: List[int],  activations_as_layers: bool = True, **kwargs):
        """
        Initializes a simple feedforward neural network in TensorFlow with ReLU activation functions.

        :param layers_dim: List of integers representing the number of neurons in each dense layer.
        :param activations_as_layers: specifies if activation functions should be defined as layers. Defaults to True.
        """
        super(TensorFlowFFNN, self).__init__(**kwargs)
        self.layers_dim = layers_dim
        self.layers_list = []
        self.activations_as_layers = activations_as_layers

        for i in range(1, len(layers_dim)):
            if self.activations_as_layers:
                self.layers_list.append(tf.keras.layers.Dense(layers_dim[i], activation=None))
                if i < len(layers_dim) - 1:
                    self.layers_list.append(tf.keras.layers.ReLU())
            else:
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
        for layer in self.layers_list: x = layer(x)
        return x

    def get_config(self):
        """
        Returns the configuration of the model for serialization.

        :return: Dictionary containing the model configuration.
        """
        config = super(TensorFlowFFNN, self).get_config()
        config.update({
            "layers_dim": self.layers_dim,
            "activations_as_layers": self.activations_as_layers,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a model instance from a configuration dictionary.

        :param config: Dictionary containing the model configuration.
        :return: An instance of TensorFlowFFNN.
        """
        return cls(**config)