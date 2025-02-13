from typing import Any, Iterator
import torch
import copy
from multipledispatch import dispatch

from NeuralNetworks.ActivationFunctions import *

@dispatch(torch.nn.Linear)
def get_layer_tensor(layer: torch.nn.Linear) -> list[list[float]]:
    """
    Extracts the weights tensor from a linear layer of a PyTorch.

    The provided tensor is the weights of the layer's linear/dense layer.
    Each weight in the tensor in position i, j represents the weight of the connection
    between the i-th neuron in the previous layer and the j-th neuron in the next layer.
    So the input dimension of a tensor t is len(t) and the output dimension is len(t[0]).

    :param layer: PyTorch linear layer
    :return: the tensor
    """
    return layer.weight.detach().numpy().transpose().tolist()


@dispatch(torch.nn.Linear)
def get_layer_biases(layer: torch.nn.Linear) -> list[float]:
    """
    Extracts the biases from a linear layer of a PyTorch.

    The provided biases are the biases of the layer's linear/dense layer.

    :param layer: PyTorch linear layer
    :return: the biases
    """
    return layer.bias.detach().numpy().tolist()


@dispatch(torch.nn.Module)
def get_layer_activation(layer: torch.nn.Module) -> ActivationFunction | None:
    """
    Extracts the activation function from a PyTorch layer.
    If the layer corresponds to a Linear layer, None is returned, otherwise if the layer corresponds to
    an activation function, the corresponding ActivationFunction object is returned.

    If the activation function layer is not supported, a NotImplementedError is raised.

    :param layer: PyTorch layer
    :return: the activation function
    """
    l_type = type(layer)

    if l_type == torch.nn.Linear:
        return None

    if l_type in {torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ReLU6}:
        max_val = 6 if l_type == torch.nn.ReLU6 else None
        threshold = 0.0
        negative_slope = layer.negative_slope if l_type == torch.nn.LeakyReLU else 0.0
        return ReLU(max_val=max_val, threshold=threshold, negative_slope=negative_slope)

    if l_type == torch.nn.Hardsigmoid:
        return HardSigmoid()

    if l_type == torch.nn.Hardtanh:
        return HardTanh(min_val=layer.min_val, max_val=layer.max_val)

    if l_type == torch.nn.Hardswish:
        return HardSwish()

    if l_type == torch.nn.Hardshrink:
        return HardShrink(lambd=layer.lambd)

    if l_type == torch.nn.Threshold:
        return Threshold(threshold=layer.threshold, value=layer.value)

    if l_type == torch.nn.Identity:
        return Identity()

    raise NotImplementedError(f"Unsupported layer type: {l_type}")


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


@dispatch(torch.nn.Module)
def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Clones a PyTorch, including its architecture and weights.

    :param model: PyTorch model
    :return: the cloned model
    """
    return copy.deepcopy(model)