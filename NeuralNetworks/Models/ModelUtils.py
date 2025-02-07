from .TorchAdapter import *
from .TFAdapter import *
from NeuralNetworks.ActivationFunctions import *

def activations(model: torch.nn.Module | tf.keras.Model) -> list[ActivationFunction]:
    """
    Extracts activation functions from a PyTorch or a TensorFlow model.
    The provided activation functions are the activation functions of the model's layers in the order
    they appear in the model from the input layer to the output layer.

    :param model: PyTorch or TensorFlow model
    :return: the list of activation functions
    """
    for layer in layers(model):
        activation = get_layer_activation(layer)
        if activation is not None:
            yield activation


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
