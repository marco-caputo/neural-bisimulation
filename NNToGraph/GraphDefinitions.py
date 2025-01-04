# Prefix strings used to define nodes in the graph based on the layer type.
I_str = "I" # Input layer
H_str = "H" # Hidden layer
B_str = "B" # Bias
O_str = "O" # Output layer

# Separators must be different
sep_1 = ""
sep_2 = "_"

def node_str(layer: str, layer_index: int = None, node_index: int = None) -> str:
    return str(layer) + \
        (f"{sep_1}{layer_index}" if layer_index is not None else "") + \
        (f"{sep_2}{node_index}" if node_index is not None else "")