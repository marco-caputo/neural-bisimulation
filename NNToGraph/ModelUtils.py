from typing import Any, Callable, Iterator

import tensorflow as tf
import torch
from multipledispatch import dispatch


AFFINE_TRANS_LAYER_TYPES = {torch.nn.Linear, tf.keras.layers.Dense}

@dispatch(torch.nn.Module)
def layers(model: torch.nn.Module, layer_type=None) -> Iterator[Any]:
    """
    Extracts layers of a specific type from a PyTorch model.

    :param model: PyTorch model
    :param layer_type: Type of the layer to extract
    :return: List of layers of the specified type
    """
    for _, layer in model.named_modules():
        if layer is None or isinstance(layer, layer_type):
            yield layer


def get_layer_tensor(layer: torch.nn.Linear | tf.keras.layers.Dense) -> list[list[float]]:
    """
    Extracts the weights tensor from a linear layer of a PyTorch or a dene layer of a TensorFlow model.

    The provided tensor is the weights of the layer's linear/dense layer.
    Each weight in the tensor in position i, j represents the weight of the connection
    between the i-th neuron in the previous layer and the j-th neuron in the next layer.
    So the input dimension of a tensor t is len(t) and the output dimension is len(t[0]).

    :param layer: PyTorch or TensorFlow layer
    :return: the tensor
    """
    if isinstance(layer, torch.nn.Linear):
        return layer.weight.detach().numpy().transpose().tolist()
    elif isinstance(layer, tf.keras.layers.Dense):
        return layer.get_weights()[0].tolist()

def get_layer_biases(layer: torch.nn.Linear | tf.keras.layers.Dense) -> list[float]:
    """
    Extracts the biases from a linear layer of a PyTorch or a dene layer of a TensorFlow model.

    The provided biases are the biases of the layer's linear/dense layer.

    :param layer: PyTorch or TensorFlow layer
    :return: the biases
    """
    if isinstance(layer, torch.nn.Linear):
        return layer.bias.detach().numpy().tolist()
    elif isinstance(layer, tf.keras.layers.Dense):
        return layer.get_weights()[1].tolist()


@dispatch(tf.keras.Model)
def layers(model: tf.keras.Model, layer_type=None) -> Iterator[Any]:
    """
    Extracts layers of a specific type from a TensorFlow model.

    :param model: TensorFlow model
    :param layer_type: Type of the layer to extract
    :return: List of layers of the specified type
    """
    for layer in model.layers:
        if layer is None or isinstance(layer, layer_type):
            yield layer


@dispatch(torch.nn.Module)
def tensors(model: torch.nn.Module) -> Iterator[list[float, float]]:
    """
    Extracts tensors from a PyTorch model as bi-dimensional lists of weights.
    The provided tensors are the weights of the model's linear layers in the order
    they appear in the model from the input layer to the output layer.

    Each weight in the tensor in position i, j represents the weight of the connection
    between the i-th neuron in the previous layer and the j-th neuron in the next layer.
    So the input dimension of a tensor t is len(t) and the output dimension is len(t[0]).

    :param model: PyTorch model
    :return: the list of tensors
    """
    for layer in layers(model, layer_type=torch.nn.Linear):
        yield get_layer_tensor(layer)


@dispatch(tf.keras.Model)
def tensors(model: tf.keras.Model) -> Iterator[list[float, float]]:
    """
    Extracts tensors from a TensorFlow model as bi-dimensional lists of weights.
    The provided tensors are the weights of the model's dense layers in the order
    they appear in the model from the input layer to the output layer.

    Each weight in the tensor in position i, j represents the weight of the connection
    between the i-th neuron in the previous layer and the j-th neuron in the next layer.
    So the input dimension of a tensor t is len(t) and the output dimension is len(t[0]).

    :param model: TensorFlow model
    :return: the list of tensors
    """
    for layer in layers(model, layer_type=tf.keras.layers.Dense):
        yield get_layer_tensor(layer)


@dispatch(torch.nn.Module)
def biases(model: torch.nn.Module) -> Iterator[list[float]]:
    """
    Extracts biases from a PyTorch model as lists.
    The provided biases are the biases of the model's linear layers in the order
    they appear in the model from the input layer to the output layer.

    :param model: PyTorch model
    :return: the list of biases
    """
    for layer in layers(model, layer_type=torch.nn.Linear):
        yield get_layer_biases(layer)


@dispatch(tf.keras.Model)
def biases(model: tf.keras.Model) -> Iterator[list[float]]:
    """
    Extracts biases from a TensorFlow model as lists.
    The provided biases are the biases of the model's dense layers in the order
    they appear in the model from the input layer to the output layer.

    :param model: TensorFlow model
    :return: the list of biases
    """
    for layer in layers(model, layer_type=tf.keras.layers.Dense):
        yield get_layer_biases(layer)


def input_dim(model: torch.nn.Module | tf.keras.Model) -> int:
    """
    Returns the input dimension of a feed-forward PyTorch or a TensorFlow model.

    :param model: PyTorch or TensorFlow model
    :return: the input dimension
    """
    return len(next(tensors(model)))


def apply_to_tensors(model: torch.nn.Module | tf.keras.Model,
                     weight_proc: Callable[[float, int, int, int], None] = None,
                     bias_proc: Callable[[float, int, int], None] = None):
    """
    Applies a procedure to all the weights and to all the biases of linear/dense layers of a model.
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

    :param model: PyTorch or TensorFlow model
    :param weight_proc: Function to apply to the weights
    :param bias_proc: Function to apply to the biases.
    """
    for layer, tensor in enumerate(tensors(model), 1):
        for i, row in enumerate(tensor):
            for j, weight in enumerate(row):
                weight_proc(weight, layer, i, j)

    if bias_proc is not None:
        for layer, bias in enumerate(biases(model), 1):
            for i, b in enumerate(bias):
                bias_proc(b, layer, i)
