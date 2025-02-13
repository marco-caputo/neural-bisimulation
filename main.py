import bispy as bp
import networkx as nx
from z3 import *

from Testing import *
from NeuralNetworks.Graphs import *
from SMTEquivalence import *


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
    x = Real('x')
    y = Real('y')
    print(get_optimal_solution(x + 2*y, [x >= 0, y >= 0, x <= 3*y, y <= 2], maximize=True))


def main():
    # bisimulation_test()
    # model_visualization_test()
    z3_test()


if __name__ == "__main__":
    main()
