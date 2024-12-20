import networkx as nx
import bispy as bp
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt

def bisimulation_test():
    dgraph = nx.DiGraph()
    dgraph.add_edge(1, 2)
    dgraph.add_edge(2, 1)
    dgraph.add_edge(2, 3)
    dgraph.add_edge(1, 4)

    print(bp.compute_maximum_bisimulation(dgraph, [(1, 2), (3, 4)]))

# Example model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def main():
    model = SimpleNN()
    graph = nx.DiGraph()

    nodes_names = []

    # Add nodes and edges with weights
    for name, param in model.named_parameters():
        print(f"name:{name} param:{param}")
        layer_name = name.split('.')[0]
        if 'weight' in name:
            i = 0
            for weight in param.data.numpy()[0]:
                i+=1
                graph.add_edge(f"{layer_name}_{i}", f"{layer_name}_{i}_out", weight=weight)

    # Visualize
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'))
    plt.show()


if __name__ == "__main__":
    main()
