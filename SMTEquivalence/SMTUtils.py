from typing import List, Tuple
from multipledispatch import dispatch
from z3 import *
import tensorflow as tf
import torch

from NeuralNetworks import NeuralNetwork


def get_float_formula_satisfiability(formula: BoolRef, inputs: List[float]) -> Tuple[bool, List[float] | None]:
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
        return True, [float(model.evaluate(x).as_decimal(10).rstrip('?')) for x in inputs] # Forse bastava fare model[x].as_long()
    elif result == unsat:
        return False, None
    else:
        raise RuntimeError("Solver returned 'unknown'. The equivalence might be too complex to decide.")


def is_satisfiable(formula: BoolRef) -> bool:
    """
    Checks the satisfiability of a Z3 formula and returns a boolean indicating if the formula is satisfiable.

    :param formula: A Z3 formula
    :return: A boolean indicating if the formula is satisfiable
    """
    s = Solver()
    s.add(formula)
    if s.check() == unknown:
        raise RuntimeError("Solver returned 'unknown'. The equivalence might be too complex to decide.")
    return s.check() == sat


def get_optimal_solution(objective: ArithRef, constraints: list[BoolRef], maximize: bool = True) -> Tuple[bool, float | None]:
    """
    Finds the optimal value of the given float objective function under the given constraints.

    The output of the function is a tuple containing a boolean indicating if the optimization problem is feasible and
    the optimal value. In particular:
    - If the optimization problem has an empty feasible region, the function returns False and None;
    - If the optimization problem is unbounded, the function returns True and None;
    - If the optimization problem is feasible and bounded, the function returns True and the optimal float value.

    :param objective: The objective function to optimize
    :param constraints: A list of constraints
    :param maximize: A boolean indicating whether to maximize or minimize the objective
    :return: A tuple containing a boolean indicating if the optimization problem is feasible and the optimal value if any
    """
    opt = Optimize()
    opt.add(And(constraints))

    obj = opt.maximize(objective) if maximize else opt.minimize(objective)
    if opt.check() == sat:
        b = opt.upper(obj) if maximize else opt.lower(obj)
        if str(b) in {"oo", "-oo"}:
            return True, None
        else:
            return True, b.as_long() if b.is_int() else b.as_decimal(10).rstrip('?')

    return False, None


def encode_into_SMT_formula(model: NeuralNetwork | tf.keras.Model | torch.nn.Module,
                            input_bounds: List[Tuple[float, float]] = None,
                            var_prefix: str = "") \
        -> Tuple[BoolRef, list[Real], list[Real]]:
    """
    Converts a NeuralNetwork, PyTorch or TensorFlow model into a SMT Formula representing its input-output relation.

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
        to represent the layer's operation. Only linear activation functions can be encoded into SMT formulas.
        Currently supported layer types and activation functions are:

        - Affine Transformation (Linear/Dense)
        - ReLU (ReLU, LeakyReLU, ReLU6)
        - HardSigmoid
        - Hard
        - HardSwish (or HardSiLU)
        - HardShrink
        - Threshold

    :param model: a NeuralNetwork, PyTorch or TensorFlow model
    :param input_bounds: A list of tuples [(l1, u1), ..., (ln, un)] specifying the inclusive lower and upper bounds
                      for each input variable.
    :param: var_prefix: A prefix to add to the names of the variables in the produced SMT formula
    :return: An SMT Formula representing the input-output relation of the model and the lists of input and output variables
    """
    if not isinstance(model, NeuralNetwork):
        model = NeuralNetwork.from_model(model)

    constraints = []

    # Encode input variables and constraints
    inputs = []
    if input_bounds is None:
        input_bounds = [(float('-inf'), float('inf'))] * model.input_size()
    if len(input_bounds) != model.input_size():
        raise ValueError(f"Expected {model.input_size()} input bounds, but got {len(input_bounds)}")

    for i, (l, u) in enumerate(input_bounds):
        x = Real(f'{var_prefix}x{i}')  # Define a Z3 variable for each input
        inputs.append(x)
        bounds = []
        if l is not None and l != float('-inf'): bounds.append(l <= x)
        if u is not None and u != float('inf'): bounds.append(x <= u)
        if len(bounds) > 0: constraints.append(And(bounds))  # Input constraints

    # Process each layer of the network
    current_layer = list(inputs)
    for l_idx, layer in enumerate(model.layers):
        next_layer = []

        # Encode the affine transformation of the layer z_j = sum_k (x_k * W_kj) + b_j
        for j in range(layer.output_size()):
            z = Real(f'{var_prefix}z{l_idx}_{j}')
            affine_expr = Sum([current_layer[k] * layer.weights[k][j] for k in range(layer.input_size())]) \
                          + layer.biases[j]
            constraints.append(simplify(z == affine_expr))
            next_layer.append(z)

        current_layer = next_layer
        next_layer = []

        # Encode the activation function of the layer h_j = f(z_j)
        for j, z in enumerate(current_layer):
            h = Real(f'{var_prefix}h{l_idx}_{j}')
            constraints.append(simplify(h == layer.activation_functions[j].formula(z)))
            next_layer.append(h)

        current_layer = next_layer

    return And(constraints), inputs, current_layer