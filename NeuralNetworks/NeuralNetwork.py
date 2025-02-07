from NeuralNetworks.ActivationFunctions import ActivationFunction, Identity
from NeuralNetworks import Layer
import torch
import tensorflow as tf
from multipledispatch import dispatch
from NeuralNetworks.Models.ModelUtils import tensors, biases, activations

class NeuralNetwork:
    layers: list[Layer]

    @dispatch(torch.nn.Module | tf.keras.Model)
    def __init__(self, model: torch.nn.Module | tf.keras.Model):
        self.__init__(list(tensors(model)), list(biases(model)), list(activations(model)))

    @dispatch(list[list[list[float]]], list[list[float]], list[ActivationFunction])
    def __init__(self, weights: list[list[list[float]]],
                 biases: list[list[float]] = None,
                 activation_functions: list[ActivationFunction] | list[list[ActivationFunction]] = None):

        if (len(weights) != len(biases) or len(weights) != len(activation_functions)):
            raise ValueError("The number of weight tensors, biases and activation functions must be the same.")

        if biases is None:
            biases = [[0] * len(weights[i]) for i in range(len(weights))]
        if activation_functions is None:
            activation_functions = [[Identity()] * len(weights[i]) for i in range(len(weights))]
        elif isinstance(activation_functions[0], ActivationFunction):
            activation_functions = [[activation_functions[i]] * len(weights[i]) for i in range(len(weights))]

        self.__init__([Layer(weights[i], biases[i], activation_functions[i]) for i in range(len(weights))])


    @dispatch(list[Layer])
    def __init__(self, layers: list[Layer]):
        for i in range(1, len(layers)):
            if layers[i-1].output_size() != layers[i].input_size():
                raise ValueError("The number of neurons in the previous layer must be equal to the number of neurons "
                                 "in the next layer.")

        self.layers = layers


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

    def __len__(self):
        return self.size()

    def __getitem__(self, index: int):
        return self.get_layer(index)

    def __iter__(self):
        return iter(self.layers)

    def __call__(self, inputs: list[float]) -> list[float]:
        return self.forward_pass(inputs)

    def __str__(self):
        return f"NeuralNetwork({self.layers})"
