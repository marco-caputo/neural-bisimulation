import bispy as bp
import networkx as nx
from z3 import *

from NeuralNetworks import create_graph, visualize_model_graph, TensorFlowFFNN, TorchFFNN


def bisimulation_test():
    dgraph = nx.DiGraph()
    dgraph.add_edge(1, 2)
    dgraph.add_edge(2, 1)
    dgraph.add_edge(2, 3)
    dgraph.add_edge(1, 4)

    print(bp.compute_maximum_bisimulation(dgraph, [(1, 2), (3, 4)]))

def model_visualization_test():
    layers_dim = [2, 3, 4, 1]  # Input: 2 neurons, Hidden1: 3 neurons, Hidden2: 4 neurons, Output: 2 neurons
    model = TorchFFNN(layers_dim)  # TensorFlowFFNN(layers_dim)
    graph = create_graph(model, add_biases=True)
    visualize_model_graph(graph, inter_layer_distance=2.0, intra_layer_distance=0.5, round_digits=4)

def z3_test():
    l = Array('l', IntSort(), RealSort())
    i: Int = Int('i')
    j: Int = Int('j')
    s = Solver()

    s.add(i <= 1)
    s.add(i >= 0)
    s.add(j == i + 1)
    s.add(l[i] <= l[j])

    print(s.check())
    print(s.model())


def main():
    # bisimulation_test()
    # model_visualization_test()
    z3_test()


if __name__ == "__main__":
    main()
