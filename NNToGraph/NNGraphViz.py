import sys
from itertools import groupby
import networkx as nx
from matplotlib import pyplot as plt
from NNToGraph import I_str, H_str, B_str, O_str, sep_1, sep_2
from typing import Dict, Tuple

NODE_SIZE = 800
NODE_COLOR = 'skyblue'
FONT_SIZE = 8
FONT_WEIGHT = 'bold'
EDGE_LABEL_FONT_COLOR = 'red'

INTER_LAYER_DISTANCE = 1.0
INTRA_LAYER_DISTANCE = 0.5
ROUND_DIGITS = 5


def visualize_model_graph(G: nx.DiGraph,
                          inter_layer_distance: float = INTER_LAYER_DISTANCE,
                          intra_layer_distance: float = INTRA_LAYER_DISTANCE,
                          round_digits: int = ROUND_DIGITS):
    """
    Visualizes a neuron-level graph representation of a neural network model. The graph is drawn with nodes
    positioned in vertical-aligned layers with the same prefix (I, H, O) and ordered in the same layer by their
    numerical suffix. The weight of the connection between neurons is displayed as an edge label.

    :param G: Neuron-level graph representation
    :param inter_layer_distance: Distance between layers
    :param intra_layer_distance: Distance between neurons in the same layer
    :param round_digits: Number of digits to round the edge weights to
    """
    pos = nn_layout(G, inter_layer_distance, intra_layer_distance)
    nx.draw(G, pos, with_labels=True, node_size=NODE_SIZE, node_color=NODE_COLOR, font_size=FONT_SIZE, font_weight=FONT_WEIGHT)
    edge_labels = nx.get_edge_attributes(G, 'weight') if round_digits is None \
        else {k: round(v, round_digits) for k, v in nx.get_edge_attributes(G, 'weight').items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color=EDGE_LABEL_FONT_COLOR)
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
        if s.startswith(H_str): return 1 + int(s[len(H_str+sep_1):])
        if s.startswith(B_str): return 1 + int(s[len(B_str+sep_1):]) - 0.5
        if s.startswith(O_str): return sys.maxsize
        return -1

    pos = {}

    # Sort nodes by their layer prefix for correct grouping
    sorted_nodes = sorted(G.nodes, key=lambda n: n.split(sep_2)[0])
    layers = [(layer_key, list(nodes)) for layer_key, nodes in groupby(sorted_nodes, key=lambda n: n.split(sep_2)[0])]

    # Sort layers by their order
    sorted_layers = sorted(layers, key=lambda x: layer_order(x[0]))

    max_nodes = max(len(nodes) for _, nodes in sorted_layers)
    i = 0  # Horizontal layer position
    for layer_key, layer_nodes in sorted_layers:
        layer_nodes = list(layer_nodes)  # Consume the iterator to make it reusable

        if layer_key.startswith(B_str): # Bias layer
            j = (max_nodes+2) * 0.5 * intra_layer_distance
            pos[layer_nodes[0]] = (i, j)

        else:
            j = len(layer_nodes) * 0.5 * intra_layer_distance # Start vertical position
            for node in sorted(layer_nodes, key=lambda x: int(x.split(sep_2)[1])):
                pos[node] = (i, j)
                j -= intra_layer_distance  # Decrement vertical position

        i += inter_layer_distance  # Increment horizontal layer position

    return pos