import copy

import torch
import torch.nn as nn
import tensorflow as tf
from typing import List

class TorchFFNN(torch.nn.Module):
    """
    A simple feedforward neural network in PyTorch with ReLU activation functions.
    """

    def __init__(self, layers_dim: List[int],
                 activations_as_layers: bool = True,
                 activation_layer: nn.Module = None,
                 output_activation: nn.Module = None):
        """
        Initializes a simple feedforward neural network in PyTorch with ReLU activation functions.

        :param layers_dim: List of integers representing the number of neurons in each linear layer.
        :param activations_as_layers: specifies if activation functions should be defined as layers. Defaults to True.
        :param activation_layer: Activation function to use, in form of a layer. Defaults to ReLU.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations_as_layers = activations_as_layers
        if activation_layer is None and activations_as_layers:
            activation_layer = torch.nn.ReLU()

        for i in range(len(layers_dim) - 1):
            if self.activations_as_layers:
                self.layers.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
                if activation_layer is not None and i < len(layers_dim) - 2:
                    self.layers.append(copy.deepcopy(activation_layer))
                elif output_activation is not None and i == len(layers_dim) - 2:
                    self.layers.append(copy.deepcopy(output_activation))
            else:
                self.layers.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))

    def forward(self, x):
        for layer in self.layers: x = layer(x) if self.activations_as_layers else torch.relu(layer(x))
        return x


class TensorFlowFFNN(tf.keras.Model):
    """
    A simple feedforward neural network in TensorFlow with ReLU activation functions
    """

    def __init__(self, layers_dim: List[int],
                 activations_as_layers: bool = True,
                 activation_func: tf.keras.layers.Layer | str = None,
                 output_activation: tf.keras.layers.Layer | str = None,
                 **kwargs):
        """
        Initializes a simple feedforward neural network in TensorFlow with ReLU activation functions.

        :param layers_dim: List of integers representing the number of neurons in each dense layer.
        :param activations_as_layers: specifies if activation functions should be defined as layers. Defaults to True.
        :param activation_func: Activation function to use, in form of a layer. Defaults to ReLU.
        """
        super(TensorFlowFFNN, self).__init__(**kwargs)
        self.layers_dim = layers_dim
        self.layers_list = []
        self.activations_as_layers = activations_as_layers
        if activation_func is None:
            activation_func = tf.keras.layers.ReLU() if activations_as_layers else 'relu'
        if output_activation is None:
            output_activation = tf.keras.layers.Identity() if activations_as_layers else None

        for i in range(1, len(layers_dim)):
            if self.activations_as_layers:
                self.layers_list.append(tf.keras.layers.Dense(layers_dim[i], activation=None))
                if i < len(layers_dim) - 1:
                    self.layers_list.append(copy.deepcopy(activation_func))
                elif output_activation is not None and i == len(layers_dim) - 1:
                    self.layers_list.append(copy.deepcopy(output_activation))
            else:
                self.layers_list.append(tf.keras.layers.Dense(layers_dim[i],
                                                              activation='relu' if i < len(layers_dim) - 1 else None))

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