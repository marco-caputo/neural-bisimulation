import networkx as nx
import tensorflow as tf
import torch
from multipledispatch import dispatch
from NNToGraph import I_str, H_str, B_str, O_str, node_str, get_layers

@dispatch(torch.nn.Module)
def create_graph(model: torch.nn.Module, add_biases: bool = True) -> nx.DiGraph:
    """
    Converts a PyTorch model into a neuron-level graph representation.
    The graph representation is a directed graph where each node represents a neuron and each edge represents a
    connection between neurons. The weight of the connection in the current model state is stored as an edge
    attribute named 'weight'.

    :param model: PyTorch model
    :param add_biases: If True, the weights and biases are accessed using the model's layers,
    otherwise the weights of the model are accessed using the Tensors in model's named parameters.
    :return: Neuron-level graph representation
    """
    G = nx.DiGraph()

    if add_biases:
        linear_layers = list(get_layers(model, layer_type=torch.nn.Linear))

        for layer_count, layer in enumerate(linear_layers, 1): # Iterate over all layers
            weights = layer.weight.detach().numpy()
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    G.add_edge(
                        node_str(I_str, node_index=j+1) if layer_count == 1 else node_str(H_str, layer_count-1, j+1),
                        node_str(O_str, node_index=i+1) if layer_count == len(linear_layers)
                        else node_str(H_str, layer_count, i+1),
                        weight=weights[i, j]
                    )

            biases = layer.bias.detach().numpy()
            if biases is not None: # Check if biases are defined
                for i in range(len(biases)): # Add biases as additional nodes
                    if biases[i] != 0: # Only add non-zero biases
                        G.add_edge(
                            node_str(B_str, layer_count),
                            node_str(O_str, node_index=i+1) if layer_count == len(linear_layers)
                            else node_str(H_str, layer_count, i+1),
                            weight=biases[i]
                        )
    else:
        tensors = [param for name, param in model.named_parameters()
                   if 'weight' in name and isinstance(param, torch.Tensor)]

        for layer_count, tensor in enumerate(tensors, 1):
            output_dim, input_dim = tensor.size()
            for i in range(input_dim):
                for j in range(output_dim):
                    G.add_edge(
                        node_str(I_str, node_index=i+1) if layer_count == 1 else node_str(H_str, layer_count-1, i+1),
                        node_str(O_str, node_index=j+1) if layer_count == len(tensors)
                        else node_str(H_str, layer_count, j+1),
                        weight=tensor[j, i].item())

    return G

@dispatch(tf.keras.Model)
def create_graph(model: tf.keras.Model, add_biases: bool = True) -> nx.DiGraph:
    """
    Converts a TensorFlow model into a neuron-level graph representation.
    The graph representation is a directed graph where each node represents a neuron and each edge represents a
    connection between neurons. The weight of the connection in the current model state is stored as an edge attribute
    named 'weight'.

    :param model: TensorFlow model
    :param add_biases: If True, both the weights and biases are accessed using the model's layers and included
    in the graph representation, otherwise only the weights are included.
    :return: Neuron-level graph representation
    """
    G = nx.DiGraph()
    dense_layers = list(get_layers(model, layer_type=tf.keras.layers.Dense))

    # Get weights of all Dense layers
    for layer_count, layer in enumerate(dense_layers, 1):
        weights, biases = layer.get_weights()  # weights: (input_dim, output_dim), biases: (output_dim)
        input_dim, output_dim = weights.shape

        for i in range(input_dim):
            for j in range(output_dim):
                G.add_edge(
                    node_str(I_str, node_index=i+1) if layer_count == 1 else node_str(H_str, layer_count-1, i+1),
                    node_str(O_str, node_index=j+1) if layer_count == len(dense_layers)
                    else node_str(H_str, layer_count, j+1),
                    weight=weights[i, j]
                )

        if add_biases:
            if biases is not None:  # Check if biases are defined
                for i in range(len(biases)):  # Add biases as additional nodes
                    if biases[i] != 0:  # Only add non-zero biases
                        G.add_edge(
                            node_str(B_str, layer_count),
                            node_str(O_str, node_index=i+1) if layer_count == len(linear_layers)
                            else node_str(H_str, layer_count, i+1),
                            weight=biases[i]
                        )

    return G
