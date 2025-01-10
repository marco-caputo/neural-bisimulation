from typing import List, Tuple, Any, Callable

from z3 import *
import tensorflow as tf
import torch
from NNToGraph import *


def are_strict_equivalent(model1: torch.nn.Module | tf.keras.Model, model2: torch.nn.Module | tf.keras.Model) \
        -> Tuple[bool, List[float] | None]:
    """
    Checks if two feed-forward models are strictly equivalent.
    Two models are strictly equivalent if for every possible input, they produce the same exact output.
    The equivalence is checked by proving the unsatisfiability of the formula that represents the negation of the
    equivalence between each of the two models output variables.
    Besides the boolean value representing the equivalence, if the two models are not strictly equivalent, a
    counterexample consisting in a list of input values that produce different outputs is returned,
    otherwise None is returned.

    Note that, in order to check the equivalence, the input and the output dimensions of the two models must be the
    same, otherwise an exception is raised.

    :param model1: The first model
    :param model2: The second model
    :return: a tuple containing a boolean indicating if the two models are strictly equivalent and a counterexample
     if they are not
    """
    if input_dim(model1) != input_dim(model2):
        raise ValueError(f"Expected models with the same input dimension, but got {input_dim(model1)} and {input_dim(model2)}")
    if output_dim(model1) != output_dim(model2):
        raise ValueError(f"Expected models with the same output dimension, but got {output_dim(model1)} and {output_dim(model2)}")

    # Encode the input-output relation of the two models into SMT formulas
    formula1, inputs1, outputs1 = encode_into_SMT_formula(model1)
    formula2, inputs2, outputs2 = encode_into_SMT_formula(model2)

    # Build the strict equivalence formula
    inputs_equivalence = And([inputs1[i] == inputs2[i] for i in range(len(inputs1))])
    outputs_equivalence = And([outputs1[i] == outputs2[i] for i in range(len(outputs2))])
    formula = Exists([*inputs1], And(inputs_equivalence, formula1, formula2, Not(outputs_equivalence)))

    # Check if the two formulas are equivalent
    return _get_formula_satisfiability(formula, inputs1)

def _get_formula_satisfiability(formula: BoolRef, inputs: List[Real]) -> Tuple[bool, List[float] | None]:
    """
    Checks the satisfiability of a Z3 formula involving the given list of real variables as inputs and returns a
    counterexample if the formula is satisfiable.

    :param formula: A Z3 formula
    :param inputs: A list of Z3 real variables involved in the formula
    :return: A tuple containing a boolean indicating if the formula is satisfiable and a counterexample if it is
    """
    s = Solver()
    s.add(formula)
    result = s.check()
    if result == sat:
        model = s.model()
        return False, [model.evaluate(x).as_decimal(5) for x in inputs]
    elif result == unsat:
        return True, None
    else:
        raise RuntimeError("Solver returned 'unknown'. The equivalence might be too complex to decide.")

def encode_into_SMT_formula(model: torch.nn.Module | tf.keras.Model, input_bounds: List[Tuple[float, float]] = None) \
        -> Tuple[BoolRef, list[Real], list[Real]]:
    """
    Converts a feed-forward model into a SMT Formula representing its input-output relation.
    In case a PyTorch model is provided, this method requires that the provided model has its activation functions
    defined as layers, in order tomake them visible and correctly encode them into the SMT formula.
    For TensorFlow models, the activation functions are not required to be defined as layers, but they must be
    activation function defined in the TensorFlow library supported by this method.

    The provided SMT Formula is a conjunction of constraints that represent the relation between the input
    and output lists of variables of the model, in particular:

    -
        The input variables are constrained to the specified inclusive input bounds, if provided.
        If specific inputs don't have a lower or upper bound, the corresponding bound can be set to None or
        float('-inf') and float('inf') respectively.
        For instance, input_bounds=[(0, 1), (None, 10), (float('-inf'), float('inf'))] adds the following constraints
        to the formula: 0 <= x1 <= 1, x2 <= 10.

    -
        Intermediate variables are introduced for each layer of the model and constraints are added to the formula
        to represent the layer's operation.
        Currently supported layer types are:

        - Affine Transformation (Linear/Dense)
        - ReLU (ReLU, LeakyReLU, ReLU6)
        - HardSigmoid
        - Hard
        - HardSwish (or HardSiLU)
        - HardShrink
        - Threshold

    :param model: a PyTorch or TensorFlow model
    :param input_bounds: A list of tuples [(l1, u1), ..., (ln, un)] specifying the inclusive lower and upper bounds
                      for each input variable.
    :return: An SMT Formula representing the input-output relation of the model and the lists of input and output variables
    """
    constraints = []

    # Encode input variables and constraints
    inputs = []
    if len(input_bounds) != input_dim(model):
        raise ValueError(f"Expected {input_dim(model)} input bounds, but got {len(input_bounds)}")
    if input_bounds is None:
        input_bounds = [(float('-inf'), float('inf'))] * input_dim(model)
    for i, (l, u) in enumerate(input_bounds):
        x = Real(f'x{i}')  # Define a Z3 variable for each input
        inputs.append(x)
        bounds = []
        if l is not None and l != float('-inf'): bounds.append(l <= x)
        if u is not None and u != float('inf'): bounds.append(x <= u)
        if len(bounds) > 0: constraints.append(And(bounds))  # Input constraints

    # Process each layer of the network
    current_layer = list(inputs)
    for l_idx, layer in enumerate(layers(model)):
        l_type, l_params = _get_layer_info(layer)
        next_layer = []

        if l_type == "AffineTrans":  # z_j = sum_k (x_k * W_kj) + b_j
            ws = l_params["tensor"]
            bs = l_params["biases"]

            for j in range(len(ws[0])):
                z = Real(f'z{l_idx}_{j}')
                affine_expr = Sum([current_layer[k] * ws[j][k] for k in range(len(ws))]) + bs[j]
                constraints.append(simplify(z == affine_expr))
                next_layer.append(z)

            if isinstance(model, tf.keras.Model) and layer.activation is not None: # Add activation function as a layer
                l_type, l_params = _get_layer_info(_to_layer(layer.activation))
                for j, z in enumerate(next_layer):
                    h = Real(f'h{l_idx}_{j}')
                    constraints.append(simplify(h == _activation_function_formula(l_type, l_params, z)))
                    next_layer[j] = h

        else:
            for j, z in enumerate(current_layer):
                h = Real(f'h{l_idx}_{j}')
                constraints.append(simplify(h == _activation_function_formula(l_type, l_params, z)))
                next_layer.append(h)

        current_layer = next_layer

    return And(constraints), inputs, current_layer


def _get_layer_info(layer: Any) -> Tuple[str, dict[str, Any]]:
    """
    Returns the type of the layer and its parameters as a dictionary.

    The parameters are specific to the layer type and are used to encode the layer into an SMT (Satisfiability Modulo Theories) formula.

    Supported layer types:

    1. **Affine Transformation** (Linear/Dense)
            z_j = sum_k(x_k * W_kj) + b_j

    2. **ReLU** (ReLU, LeakyReLU, ReLU6)
           h_j = min(**max_val**, z_j) if z_j >= **threshold**,
                 **negative_slope** * (z_j - **threshold**) otherwise

    3. **HardSigmoid**
            h_j = max(0, min(1, z_j / 6 + 0.5))

    4. **HardTanh**
            h_j = max(**min_val**, min(**max_val**, z_j))

    5. **HardSwish** (or HardSiLU)
            h_j = z_j * max(0, min(1, z_j / 6 + 0.5))

    6. **HardShrink**
           h_j = z_j if z_j > **lambda** or z_j < -**lambda**,
                 0 otherwise

    7. **Threshold**
           h_j = z_j if z_j > **threshold**,
                 **<value otherwise>**

    :param layer: A PyTorch or TensorFlow layer
    :return: A tuple containing the layer type as a string and its parameters as a dictionary
    """
    l_type = type(layer)

    if l_type in AFFINE_TRANS_LAYER_TYPES:
        return "AffineTrans", {"tensor": get_layer_tensor(layer), "biases": get_layer_biases(layer)}

    if l_type in [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ReLU6, tf.keras.layers.ReLU, tf.keras.layers.LeakyRelu] or \
            (l_type == tf.keras.layers.Activation and layer.activation in
             [tf.keras.activations.relu, tf.keras.activations.leaky_relu, tf.keras.activations.relu6]):
        max_val = layer.max_val if l_type == tf.keras.layers.ReLU else \
            layer.activation.max_val if l_type == tf.keras.layers.Activation and layer.activation == tf.keras.activations.relu else \
                6 if l_type == torch.nn.ReLU6 or (
                            l_type == tf.keras.layers.Activation and layer.activation == tf.keras.activations.relu6) else \
                    float("inf")

        threshold = layer.threshold if l_type in [tf.keras.layers.ReLU] else \
            layer.activation.threshold if l_type == tf.keras.layers.Activation and layer.activation == tf.keras.activations.relu else \
                0.0

        negative_slope = layer.negative_slope if l_type in [torch.nn.LeakyReLU, tf.keras.layers.ReLU,
                                                            tf.keras.layers.LeakyRelu] else \
            layer.activation.negative_slope if l_type == tf.keras.layers.Activation else \
                0.0

        return "ReLU", {"max_val": max_val, "threshold": threshold, "negative_slope": negative_slope}

    if l_type == torch.nn.Hardsigmoid or \
            (l_type == tf.keras.layers.Activation and layer.activation == tf.keras.activations.hard_sigmoid):
        return "HardSigmoid", {}

    if l_type == torch.nn.Hardtanh:
        return "HardTanh", {"min_val": layer.min_val, "max_val": layer.max_val}

    if l_type == torch.nn.Hardswish or \
            (l_type == tf.keras.layers.Activation and layer.activation in
             [tf.keras.activations.hard_swish, tf.keras.activations.hard_silu]):
        return "HardSwish", {}

    if l_type == torch.nn.Hardshrink:
        return "HardShrink", {"lambda": layer.lambd}

    if l_type == torch.nn.Threshold:
        return "Threshold", {"threshold": layer.threshold, "value": layer.value}

    if l_type in [torch.nn.Identity, tf.keras.layers.Identity]:
        return "Identity", {}

    raise NotImplementedError(f"Unsupported layer type: {l_type}")


def _to_layer(activation: Callable) -> Any:
    """
    Converts an tensor flow activation function into a layer, if not already a layer.

    :param activation: The activation function
    :return: The corresponding layer
    """
    if type(activation) in [tf.keras.layers.ReLU, tf.keras.layers.LeakyReLU,
                            tf.keras.layers.ReLU6, tf.keras.layers.Activation]:
        return activation

    if activation == tf.keras.activations.relu:
        return tf.keras.layers.ReLU()
    if activation == tf.keras.activations.leaky_relu:
        return tf.keras.layers.LeakyReLU()
    if activation == tf.keras.activations.relu6:
        return tf.keras.layers.ReLU(6)
    if activation == tf.keras.activations.hard_sigmoid:
        return tf.keras.layers.Activation(tf.keras.activations.hard_sigmoid)
    if activation == tf.keras.activations.hard_swish:
        return tf.keras.layers.Activation(tf.keras.activations.hard_swish)
    if activation == tf.keras.activations.hard_silu:
        return tf.keras.layers.Activation(tf.keras.activations.hard_silu)

    raise NotImplementedError(f"Unsupported activation function: {activation}")


def _activation_function_formula(type: str, p: dict[str, Any], z: Real) -> BoolRef:
    """
    Returns the SMT formula that represents the activation function of a layer.

    :param type: The type of the layer
    :param p: The parameters of the layer
    :param z: The input variable of the layer
    :return: The SMT formula representing the activation function
    """
    if type == "ReLU":  # h_j = min(max_val, z_j) if z_j >= threshold, negative_slope * (z_j - threshold) otherwise
        return If(z >= p["max_val"], p["max_val"],
                  If(z >= p["threshold"], z,
                     p["negative_slope"] * (z - p["threshold"])))

    if type == "HardSigmoid":  # h_j = max(0, min(1, z_j / 6 + 0.5))
        return If(z <= -3, 0,
                  If(z >= 3, 1,
                     z / 6 + 0.5))

    if type == "HardTanh":  # h_j = max(min_val, min(max_val, z_j))
        return If(z <= p["min_val"], p["min_val"],
                  If(z >= p["max_val"], p["max_val"],
                     z))

    if type == "HardSwish":  # h_j = z_j * max(0, min(1, z_j / 6 + 0.5))
        return z * \
                    If(z <= -3, 0,
                        If(z >= 3, 1,
                            z / 6 + 0.5))

    if type == "HardShrink":  # h_j = z_j if z_j > lambda  or z_j < -lambda, 0 otherwise
        return If(Or(z > p["lambda"], z < -p["lambda"]), z, 0)

    if type == "Threshold":  # h_j = z_j if z_j > threshold, value otherwise
        return If(z > p["threshold"], z, p["value"])

    if type == "Identity":
        return z

    raise NotImplementedError(f"Unsupported layer type: {type}")
