import sys

import bispy as bp
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from itertools import groupby
from typing import List, Iterable, Tuple

I_str = "I"
H_str = "H"
O_str = "O"

def bisimulation_test():
    dgraph = nx.DiGraph()
    dgraph.add_edge(1, 2)
    dgraph.add_edge(2, 1)
    dgraph.add_edge(2, 3)
    dgraph.add_edge(1, 4)

    print(bp.compute_maximum_bisimulation(dgraph, [(1, 2), (3, 4)]))

# Example model
class SimpleFFNN(nn.Module):
    def __init__(self, layers_dim: List[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_dim) - 1):
            self.layers.append(nn.Linear(layers_dim[i], layers_dim[i+1]))

    def forward(self, x):
        for layer in self.layers: x = torch.relu(layer(x))
        return x

def create_graph(model : torch.nn.Module):
    """
    Converts a PyTorch model into a neuron-level graph representation.
    Each neuron is a node, and edges are labeled with weights.
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
                    f"{I_str}_{i+1}" if layer_count == 1 else f"{H_str}{layer_count-1}_{i+1}",
                    f"{O_str}_{j+1}" if layer_count == len(tensors) else f"{H_str}{layer_count}_{j+1}",
                    weight=tensor[j, i].item())
    return G

def visualize_model_graph(G):
    pos = nn_layout(G)
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    plt.show()

def nn_layout(G):
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
        j = -len(layer_nodes)/4  # Start vertical position
        for node in sorted(layer_nodes, key=lambda x: int(x.split("_")[1])):
            pos[node] = (i, j)
            j += 0.5  # Increment vertical position
        i += 1  # Increment horizontal layer position

    return pos

def main():
    model = SimpleFFNN([2, 3, 1])
    graph = create_graph(model)
    visualize_model_graph(graph)


if __name__ == "__main__":
    main()
