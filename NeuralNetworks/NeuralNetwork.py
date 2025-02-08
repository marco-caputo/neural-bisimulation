import copy
from typing import Callable

from NeuralNetworks.ActivationFunctions import ActivationFunction, Identity
from NeuralNetworks.Layer import Layer
import torch
import tensorflow as tf
from NeuralNetworks.Models.ModelUtils import tensors, biases, activations

class NeuralNetwork:
    layers: list[Layer]

    @classmethod
    def from_model(cls, model: torch.nn.Module | tf.keras.Model):
        """
        Initializes a neural network from a PyTorch or TensorFlow model.

        In case a PyTorch model is provided, this constructor requires that the provided model has its activation
        functions defined as layers, in order tomake them visible and correctly encode them into the NeuralNetwork
        object.
        For TensorFlow models, the activation functions are not required to be defined as layers, however, they must
        be TensorFlow activations supported by this library.

        :param model: the PyTorch or TensorFlow model
        """
        return cls(list(tensors(model)), list(biases(model)), list(activations(model)))

    @classmethod
    def from_layers(cls, layers: list[Layer]):
        """
        Initializes a neural network from a list of layers.

        :param layers: the list of layers
        """
        return cls([layer.weights for layer in layers],
                   [layer.biases for layer in layers],
                   [layer.activation_functions for layer in layers])

    def __init__(self, weights: list[list[list[float]]],
                 biases: list[list[float]] = None,
                 activation_functions: list[ActivationFunction] | list[list[ActivationFunction]] = None):

        if len(weights) != len(biases) or len(weights) != len(activation_functions):
            raise ValueError("The number of weight tensors, biases and activation functions must be the same.")

        if biases is None:
            biases = [[0] * len(weights[i][0]) for i in range(len(weights))]
        if activation_functions is None:
            activation_functions = [[Identity()] * len(weights[i][0]) for i in range(len(weights))]
        elif isinstance(activation_functions[0], ActivationFunction):
            activation_functions = [[activation_functions[i]] * len(weights[i][0]) for i in range(len(weights))]

        self.layers = [Layer(weights[i], biases[i], activation_functions[i]) for i in range(len(weights))]
        for i in range(1, len(self.layers)):
            if self.layers[i-1].output_size() != self.layers[i].input_size():
                raise ValueError("The number of neurons in the previous layer must be equal to the number of neurons "
                                 "in the next layer.")


    def size(self):
        """
        Returns the number of layers in the neural network (except the input layer).

        :return: the number of layers
        """
        return len(self.layers)

    def input_size(self):
        """
        Returns the size of the input layer of the neural network.
        This size is equal to the number of neurons in the input layer.

        :return: the size of the input layer
        """
        return self.layers[0].input_size()

    def output_size(self):
        """
        Returns the size of the output layer of the neural network.
        This size is equal to the number of neurons in the output layer.

        :return: the size of the output layer
        """
        return self.layers[-1].output_size()

    def get_layer(self, index: int):
        """
        Returns the weights, biases and activation functions of the layer at the given index.
        The given index must be in the range [0, size()).

        :param index: the index of the layer
        :return: the layer at the given index
        """
        return self.layers[index]

    def forward_pass(self, inputs: list[float]) -> list[float]:
        """
        Computes the output of the neural network given the input.

        :param inputs: the input to the neural network
        :return: the output of the neural network
        """
        if len(inputs) != self.input_size():
            raise ValueError("The number of inputs must be equal to the number of neurons in the input layer.")

        for layer in self.layers:
            inputs = layer.forward_pass(inputs)

        return inputs

    def apply_to_tensors(self,
                         weight_proc: Callable[[float, int, int, int], None] = None,
                         bias_proc: Callable[[float, int, int], None] = None):
        """
        Applies a procedure to all the weights and to all the biases of linear/dense layers of the neural network.
        The procedures are applied to the weights and biases in the order they appear in the model.
        Every weights and bias is associated with the layer of neurons it is used in.
        For instance, layer one weights are those connecting the input layer to the first hidden layer.

        Layers are 1-indexed while rows and columns are 0-indexed.

        The provided functions must have the following signature:
        - weight_proc: (Float,Int,Int,Int) -> None; where the arguments are respectively:
        weight value, layer index i, index of neuron in layer i-1, index of neuron in layer i.
        - bias_proc: (Float,Int,Int) -> None; where the arguments are respectively:
        bias value, layer index, index of neuron in layer i.

        Both procedures are optional and, if both are provided, the bias procedure is applied after all weights
        have been processed.

        :param weight_proc: Function to apply to the weights
        :param bias_proc: Function to apply to the biases.
        """
        if weight_proc is not None:
            for idx, layer in enumerate(self.layers, 1):
                for i, row in enumerate(layer.weights):
                    for j, weight in enumerate(row):
                        weight_proc(weight, idx, i, j)

        if bias_proc is not None:
            for idx, layer in enumerate(self.layers, 1):
                for i, b in enumerate(layer.biases):
                    bias_proc(b, idx, i)

    def __len__(self):
        return self.size()

    def __getitem__(self, index: int):
        return self.get_layer(index)

    def __iter__(self):
        return iter(self.layers)

    def __call__(self, inputs: list[float]) -> list[float]:
        return self.forward_pass(inputs)

    def __deepcopy__(self, memodict={}):
        return NeuralNetwork.from_layers([copy.deepcopy(layer) for layer in self.layers])

    def __str__(self):
        return f"NeuralNetwork({self.layers})"
