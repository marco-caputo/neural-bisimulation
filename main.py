import bispy as bp
import networkx as nx

from NNToGraph import create_graph, visualize_model_graph, TensorFlowFFNN, TorchFFNN


def bisimulation_test():
    dgraph = nx.DiGraph()
    dgraph.add_edge(1, 2)
    dgraph.add_edge(2, 1)
    dgraph.add_edge(2, 3)
    dgraph.add_edge(1, 4)

    print(bp.compute_maximum_bisimulation(dgraph, [(1, 2), (3, 4)]))


def main():
    layers_dim = [2, 3, 4, 1]  # Input: 2 neurons, Hidden1: 3 neurons, Hidden2: 4 neurons, Output: 2 neurons
    model = TensorFlowFFNN(layers_dim) #TorchFFNN(layers_dim)
    graph = create_graph(model)
    visualize_model_graph(graph, inter_layer_distance=2.0, intra_layer_distance=0.5, round_digits=4)


if __name__ == "__main__":
    main()
