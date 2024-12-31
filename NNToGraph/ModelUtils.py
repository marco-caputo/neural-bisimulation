from typing import Generator, Any

import tensorflow as tf
import torch
from multipledispatch import dispatch


@dispatch(torch.nn.Module)
def get_layers(model: torch.nn.Module, layer_type=None) -> Generator[Any, None, None]:
    """
    Extracts layers of a specific type from a PyTorch model.

    :param model: PyTorch model
    :param layer_type: Type of the layer to extract
    :return: List of layers of the specified type
    """
    for _, layer in model.named_modules():
        if layer is None or isinstance(layer, layer_type):
            yield layer


@dispatch(tf.keras.Model, layer_type=None)
def get_layers(model: tf.keras.Model, layer_type=None) -> Generator[Any, None, None]:
    """
    Extracts layers of a specific type from a TensorFlow model.

    :param model: TensorFlow model
    :param layer_type: Type of the layer to extract
    :return: List of layers of the specified type
    """
    for layer in model.layers:
        if isinstance(layer, layer_type):
            yield layer
