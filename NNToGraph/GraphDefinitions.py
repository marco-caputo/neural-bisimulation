I_str = "I"
H_str = "H"
B_str = "B"
O_str = "O"

def node_str(layer: str, layer_index: int = None, node_index: int = None) -> str:
    return str(layer) + \
        (str(layer_index) if layer_index is not None else "") + \
        (f"_{node_index}" if node_index is not None else "")