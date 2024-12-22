import networkx as nx
import tensorflow as tf
import torch
from multipledispatch import dispatch

I_str = "I"
H_str = "H"
O_str = "O"

@dispatch(torch.nn.Module)
def create_graph(model: torch.nn.Module) -> nx.DiGraph:
    """
    Converts a PyTorch model into a neuron-level graph representation.
    The graph representation is a directed graph where each node represents a neuron and each edge represents a
    connection between neurons. The weight of the connection in the current model state is stored as an edge attribute
    named 'weight'.

    :param model: PyTorch model
    :return: Neuron-level graph representation
    """
    G = nx.DiGraph()
    layer_count = 0

    tensors = [param for name, param in model.named_parameters()
               if 'weight' in name and isinstance(param, torch.Tensor)]

    for tensor in tensors:
        layer_count += 1
        output_dim, input_dim = tensor.size()
        for i in range(input_dim):
            for j in range(output_dim):
                G.add_edge(
                    f"{I_str}_{i + 1}" if layer_count == 1 else f"{H_str}{layer_count - 1}_{i + 1}",
                    f"{O_str}_{j + 1}" if layer_count == len(tensors) else f"{H_str}{layer_count}_{j + 1}",
                    weight=tensor[j, i].item())
    return G

@dispatch(tf.keras.Model)
def create_graph(model: tf.keras.Model) -> nx.DiGraph:
    """
    Converts a TensorFlow model into a neuron-level graph representation.
    The graph representation is a directed graph where each node represents a neuron and each edge represents a
    connection between neurons. The weight of the connection in the current model state is stored as an edge attribute
    named 'weight'.

    :param model: TensorFlow model
    :return: Neuron-level graph representation
    """
    G = nx.DiGraph()
    layer_count = 0

    # Get weights of all Dense layers
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]

    for layer in dense_layers:
        layer_count += 1
        weights, biases = layer.get_weights()  # weights: (input_dim, output_dim), biases: (output_dim,)
        input_dim, output_dim = weights.shape

        for i in range(input_dim):
            for j in range(output_dim):
                G.add_edge(
                    f"{I_str}_{i + 1}" if layer_count == 1 else f"{H_str}{layer_count - 1}_{i + 1}",
                    f"{O_str}_{j + 1}" if layer_count == len(dense_layers) else f"{H_str}{layer_count}_{j + 1}",
                    weight=weights[i, j]
                )
    return G
