import sys
from itertools import groupby
import networkx as nx
from matplotlib import pyplot as plt
from NNToGraph import I_str, H_str, O_str
from typing import Dict, Tuple


def visualize_model_graph(G: nx.DiGraph, inter_layer_distance: float = 1.0, intra_layer_distance: float = 0.5,
                          round_digits: int = None):
    """
    Visualizes a neuron-level graph representation of a neural network model. The graph is drawn with nodes
    positioned in vertical-aligned layers with the same prefix (I, H, O) and ordered in the same layer by their
    numerical suffix. The weight of the connection between neurons is displayed as an edge label.

    :param G: Neuron-level graph representation
    :param inter_layer_distance: Distance between layers
    :param intra_layer_distance: Distance between neurons in the same layer
    """
    pos = nn_layout(G, inter_layer_distance, intra_layer_distance)
    nx.draw(G, pos, with_labels=True, node_size=800, node_color='skyblue', font_size=8, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight') if round_digits is None \
        else {k: round(v, round_digits) for k, v in nx.get_edge_attributes(G, 'weight').items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.show()


def nn_layout(G: nx.DiGraph, inter_layer_distance: float, intra_layer_distance: float) -> Dict[str, Tuple[float, float]]:
    """
    Computes the layout of a neuron-level graph representation of a neural network model. The layout is a dictionary
    where the key is the node name and the value is a tuple of the x and y coordinates of the node.
    Nodes are positioned in vertical-aligned layers with the same prefix (I, H, O) and ordered in the same
    layer by their numerical suffix.

    :param G: Neuron-level graph representation
    :param inter_layer_distance: Distance between layers
    :param intra_layer_distance: Distance between neurons in the same layer
    :return: Node layout
    """

    # Sort layers by their order I -> H1 -> H2 -> ... -> O
    def layer_order(s: str):
        if s.startswith(I_str): return 0
        if s.startswith(H_str): return 1 + int(s[len(H_str):])
        if s.startswith(O_str): return sys.maxsize
        return -1

    pos = {}

    # Sort nodes by their layer prefix for correct grouping
    sorted_nodes = sorted(G.nodes, key=lambda n: n.split("_")[0])
    layers = [(layer_key, list(nodes)) for layer_key, nodes in groupby(sorted_nodes, key=lambda n: n.split("_")[0])]

    # Sort layers by their order
    sorted_layers = sorted(layers, key=lambda x: layer_order(x[0]))

    i = 0  # Horizontal layer position
    for _, layer_nodes in sorted_layers:
        layer_nodes = list(layer_nodes)  # Consume the iterator to make it reusable
        j = -len(layer_nodes) * 0.5 * intra_layer_distance  # Start vertical position
        for node in sorted(layer_nodes, key=lambda x: int(x.split("_")[1])):
            pos[node] = (i, j)
            j += intra_layer_distance  # Increment vertical position
        i += inter_layer_distance  # Increment horizontal layer position

    return pos