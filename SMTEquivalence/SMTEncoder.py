from typing import List, Tuple

from z3 import *
import tensorflow as tf
import torch
from multipledispatch import dispatch

@dispatch(torch.nn.Module)
def create_graph(model: torch.nn.Module, input_bounds: List[Tuple[Int, Int]] = None) -> BoolRef:
    """
    Converts a PyTorch model into a SMT Formula representing its input-output relation.
    The SMT Formula is a conjunction of constraints that represent the input-output relation of the model,
    in particular:


    :param model: PyTorch model
    :param input_bounds: A list of tuples [(l1, u1), ..., (ln, un)] specifying the lower and upper bounds
                      for each input variable.
    :return: SMT Formula
    """

    variables = {}
    constraints = []

    # Encode input variables and constraints
    inputs = []


    for i, (l, u) in enumerate(input_bounds):
        x = Real(f'x{i + 1}')  # Define a Z3 variable for each input
        variables[f'x{i + 1}'] = x
        inputs.append(x)
        constraints.append(And(l <= x, x <= u))  # Input constraints

    # Process each layer of the network
    current_layer = inputs
    for layer_idx, layer in enumerate(model):
        if isinstance(layer, torch.nn.Linear):
            # Affine transformation: z_j = sum_k (x_k * W_kj) + b_j
            weights = layer.weight.detach().numpy()
            biases = layer.bias.detach().numpy()
            next_layer = []

            for j in range(weights.shape[0]):
                z = Real(f'z{layer_idx}_{j}')
                variables[f'z{layer_idx}_{j}'] = z

                affine_expr = Sum([current_layer[k] * weights[j][k] for k in range(len(current_layer))]) + biases[j]
                constraints.append(z == affine_expr)
                next_layer.append(z)
            current_layer = next_layer

        elif isinstance(layer, torch.nn.ReLU):
            # ReLU activation: z_j >= 0 -> h_j = z_j; z_j < 0 -> h_j = 0
            next_layer = []
            for j, z in enumerate(current_layer):
                h = Real(f'h{layer_idx}_{j}')
                variables[f'h{layer_idx}_{j}'] = h
                constraints.append(Or(And(z >= 0, h == z), And(z < 0, h == 0)))
                next_layer.append(h)
            current_layer = next_layer

        else:
            raise NotImplementedError(f"Unsupported layer type: {type(layer)}")

    pass