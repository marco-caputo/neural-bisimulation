from typing import Any, Callable, Iterator
import copy
import tensorflow as tf
import torch
from multipledispatch import dispatch


AFFINE_TRANS_LAYER_TYPES = {torch.nn.Linear, tf.keras.layers.Dense}

@dispatch(torch.nn.Module)
def layers(model: torch.nn.Module, layer_type=None) -> Iterator[Any]:
    """
    Extracts layers of a specific type from all ModuleLists within a PyTorch model.
    If no layer type is specified, all layers are extracted.

    :param model: PyTorch model
    :param layer_type: Type of the layer to extract
    :return: List of layers of the specified type
    """
    for _, module in model.named_children():
        if isinstance(module, torch.nn.ModuleList):
            for layer in module:
                if layer_type is None or isinstance(layer, layer_type):
                    yield layer


def get_layer_tensor(layer: torch.nn.Linear | tf.keras.layers.Dense) -> list[list[float]]:
    """
    Extracts the weights tensor from a linear layer of a PyTorch or a dense layer of a TensorFlow model.

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
    Extracts the biases from a linear layer of a PyTorch or a dense layer of a TensorFlow model.

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
        if layer_type is None or isinstance(layer, layer_type):
            yield layer


@dispatch(torch.nn.Module)
def tensors(model: torch.nn.Module) -> Iterator[list[list[float]]]:
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
def tensors(model: tf.keras.Model) -> Iterator[list[list[float]]]:
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

def output_dim(model: torch.nn.Module | tf.keras.Model) -> int:
    """
    Returns the output dimension of a feed-forward PyTorch or a TensorFlow model.

    :param model: PyTorch or TensorFlow model
    :return: the output dimension
    """
    return len(list(tensors(model))[-1][0])


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


def set_weights_on_layer(layer: torch.nn.Linear | tf.keras.layers.Dense,
                         weights: list[list[float]],
                         biases: list[float]):
    """
    Sets the weights and biases of a PyTorch or TensorFlow linear/dense layer.

    :param layer: PyTorch (torch.nn.Linear) or TensorFlow (tf.keras.layers.Dense) layer
    :param weights: List of lists representing the weight matrix
    :param biases: List representing the bias vector
    """
    if len(get_layer_tensor(layer)) != len(weights) or len(get_layer_tensor(layer)[0]) != len(weights[0]):
        raise ValueError("The provided weights have a different shape than the layer's weight matrix.")
    if len(get_layer_biases(layer)) != len(biases):
        raise ValueError("The provided biases have a different shape than the layer's bias vector.")

    if isinstance(layer, torch.nn.Linear):
        with torch.no_grad():
            layer.weight = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32).transpose(0, 1))
            layer.bias = torch.nn.Parameter(torch.tensor(biases, dtype=torch.float32))
    elif isinstance(layer, tf.keras.layers.Dense):
        layer.set_weights([
            tf.convert_to_tensor(weights, dtype=tf.float32),
            tf.convert_to_tensor(biases, dtype=tf.float32)
        ])


@dispatch(torch.nn.Module)
def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Clones a PyTorch, including its architecture and weights.

    :param model: PyTorch model
    :return: the cloned model
    """
    return copy.deepcopy(model)


@dispatch(tf.keras.Model)
def clone_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Clones a TensorFlow model, including its architecture and weights.

    :param model: TensorFlow model
    :return: the cloned model
    """
    copy = tf.keras.models.clone_model(model)
    copy.set_weights(model.get_weights())
    return copy
