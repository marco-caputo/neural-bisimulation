from typing import Any, Iterator, Callable
import tensorflow as tf
import torch
from multipledispatch import dispatch

from NeuralNetworks.ActivationFunctions import *

@dispatch(tf.keras.layers.Dense)
def get_layer_tensor(layer: tf.keras.layers.Dense) -> list[list[float]]:
    """
    Extracts the weights tensor from a linear layer of a dense layer of a TensorFlow model.

    The provided tensor is the weights of the layer's linear/dense layer.
    Each weight in the tensor in position i, j represents the weight of the connection
    between the i-th neuron in the previous layer and the j-th neuron in the next layer.
    So the input dimension of a tensor t is len(t) and the output dimension is len(t[0]).

    :param layer: TensorFlow dense layer
    :return: the tensor
    """
    return layer.get_weights()[0].tolist()


@dispatch(tf.keras.layers.Dense)
def get_layer_biases(layer: tf.keras.layers.Dense) -> list[float]:
    """
    Extracts the biases from a dense layer of a TensorFlow model.

    The provided biases are the biases of the layer's linear/dense layer.

    :param layer: TensorFlow dense layer
    :return: the biases
    """
    return layer.get_weights()[1].tolist()


@dispatch(tf.keras.layers.Layer)
def get_layer_activation(layer: tf.keras.layers.Layer) -> ActivationFunction | None:
    """
    Extracts the activation function from a TensorFlow layer.
    If the layer corresponds to a Dense layer, None is returned, otherwise if the layer corresponds to
    an activation function, the corresponding ActivationFunction object is returned.

    If the activation function layer is not supported, a NotImplementedError is raised.

    :param layer: TensorFlow layer
    :return: the activation function
    """
    l_type = type(layer)

    if l_type in {tf.keras.layers.Dense, tf.keras.layers.Activation} and \
            layer.activation not in {None, tf.keras.activations.linear}:
        layer = _to_layer(layer.activation)
        l_type = type(layer)

    if l_type == tf.keras.layers.Dense:
        return None

    if l_type in {tf.keras.layers.ReLU, tf.keras.layers.LeakyReLU} or \
            (l_type == tf.keras.layers.Activation and layer.activation in
             {tf.keras.activations.relu, tf.keras.activations.relu6}):
        max_val = layer.max_value if l_type == tf.keras.layers.ReLU else (
            6 if l_type == tf.keras.layers.Activation and layer.activation == tf.keras.activations.relu6 else
            None)
        threshold = layer.threshold if l_type == tf.keras.layers.ReLU else 0.0
        negative_slope = layer.negative_slope if l_type in {tf.keras.layers.ReLU, tf.keras.layers.LeakyReLU} else (
            0.2 if l_type == tf.keras.layers.Activation and layer.activation == tf.keras.activations.leaky_relu else
            0.0)
        return ReLU(max_val=max_val, threshold=threshold, negative_slope=negative_slope)

    if l_type == tf.keras.layers.Activation and layer.activation == tf.keras.activations.hard_sigmoid:
        return HardSigmoid()

    if (l_type == tf.keras.layers.Activation and layer.activation in
            {tf.keras.activations.hard_swish, tf.keras.activations.hard_silu}):
        return HardSwish()

    if l_type == tf.keras.layers.Identity:
        return Identity()

    raise NotImplementedError(f"Unsupported layer type: {l_type}")


def _to_layer(activation: Callable) -> Any:
    """
    Converts an tensor flow activation function into a layer, if not already a layer.

    :param activation: The activation function
    :return: The corresponding layer
    """
    if activation == tf.keras.activations.relu:
        return tf.keras.layers.ReLU()
    if activation == tf.keras.activations.leaky_relu:
        return tf.keras.layers.LeakyReLU()
    if activation == tf.keras.activations.relu6:
        return tf.keras.layers.ReLU(6)
    if activation == tf.keras.activations.hard_sigmoid:
        return tf.keras.layers.Activation(tf.keras.activations.hard_sigmoid)
    if activation == tf.keras.activations.hard_swish:
        return tf.keras.layers.Activation(tf.keras.activations.hard_swish)
    if activation == tf.keras.activations.hard_silu:
        return tf.keras.layers.Activation(tf.keras.activations.hard_silu)

    raise NotImplementedError(f"Unsupported activation function: {activation}")


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