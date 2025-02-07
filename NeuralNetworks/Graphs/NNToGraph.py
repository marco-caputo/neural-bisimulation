from typing import Callable
import networkx as nx
import tensorflow as tf
import torch

from NeuralNetworks import tensors, apply_to_tensors
from NeuralNetworks.Graphs import I_str, H_str, B_str, O_str, node_str

def _add_weight_edge_proc(G: nx.DiGraph, num_of_layers: int) -> Callable[[float, int, int, int], None]:
    def add_weight_edge(weight: float, layer: int, i: int, j: int) -> None:
        G.add_edge(
            node_str(I_str, node_index=i+1) if layer == 1 else node_str(H_str, layer-1, i+1),
            node_str(O_str, node_index=j+1) if layer == num_of_layers else node_str(H_str, layer, j+1),
            weight=weight
        )
    return add_weight_edge

def _add_bias_edge_proc(G: nx.DiGraph, num_of_layers: int) -> Callable[[float, int, int], None]:
    def add_bias_edge(bias: float, layer: int, i: int) -> None:
        if bias != 0:
            G.add_edge(
                node_str(B_str, layer),
                node_str(O_str, node_index=i+1) if layer == num_of_layers else node_str(H_str, layer, i+1),
                weight=bias
            )
    return add_bias_edge

def create_graph(model: torch.nn.Module | tf.keras.Model, add_biases: bool = True) -> nx.DiGraph:
    """
    Converts a model into a neuron-level graph representation.
    The graph representation is a directed graph where each node represents a neuron and each edge represents a
    connection between neurons. The weight of the connection in the current model state is stored as an edge
    attribute named 'weight'.

    :param model: a PyTorch or TensorFlow model,
    :param add_biases: If True, the weights and biases are accessed using the model's layers,
    otherwise the weights of the model are accessed using the Tensors in model's named parameters.
    :return: Neuron-level graph representation
    """
    G = nx.DiGraph()
    num_of_layers = len(list(tensors(model)))+1
    apply_to_tensors(model,
                     _add_weight_edge_proc(G, num_of_layers),
                     _add_bias_edge_proc(G, num_of_layers) if add_biases else None)
    return G
