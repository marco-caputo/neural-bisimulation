from typing import List, Tuple
from z3 import *

from .SMTUtils import get_float_formula_satisfiability
from NeuralNetworks import *


def convert_models_to_neural_networks(model1: torch.nn.Module | tf.keras.Model,
                                      model2: torch.nn.Module | tf.keras.Model) -> Tuple[NeuralNetwork, NeuralNetwork]:
    """
    Converts two models, possibly from PyTorch or TensorFlow, into NeuralNetwork objects.
    If the models are already NeuralNetwork objects, they are returned as they are.
    """
    if not isinstance(model1, NeuralNetwork):
        model1 = NeuralNetwork.from_model(model1)
    if not isinstance(model2, NeuralNetwork):
        model2 = NeuralNetwork.from_model(model2)
    return model1, model2


def encode_into_SMT_formula(model: NeuralNetwork | tf.keras.Model | torch.nn.Module,
                            input_bounds: List[Tuple[float, float] | List[Tuple[float, float]]] = None,
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
        - Sigmoid (Approximated)
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

    # Encode input bounds
    def get_bounds(x: Real, l: float, u: float):
        b = []
        if l is not None and l != float('-inf'): b.append(l <= x)
        if u is not None and u != float('inf'): b.append(x <= u)
        return And(b)

    for i, lu_bounds in enumerate(input_bounds):
        x = Real(f'{var_prefix}x{i}')  # Define a Z3 variable for each input
        inputs.append(x)
        bounds_list = []

        if isinstance(lu_bounds, tuple):
            bounds_list.append(get_bounds(x, *lu_bounds))
        else:
            single_variable_bounds = []
            for bounds in lu_bounds: single_variable_bounds.append(get_bounds(x, *bounds))
            bounds_list.append(Or(single_variable_bounds))

        if len(bounds_list) > 0:
            constraints.append(And(bounds_list))

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


def are_strict_equivalent(model1: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                          model2: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                          input_bounds: List[Tuple[float, float] | List[Tuple[float, float]]] | None = None,
                          epsilon: float = 1e-6) \
        -> Tuple[bool, List[float] | None]:
    """
    Checks if two feed-forward models are strictly equivalent.
    Two models are strictly equivalent if for every possible input, they produce the same output (within a given epsilon
    tolerance due to floating-point arithmetic).

    The equivalence is checked by proving the unsatisfiability of the formula that represents the negation of the
    equivalence between each of the two models output variables.
    Besides the boolean value representing the equivalence, if the two models are not strictly equivalent, a
    counterexample consisting in a list of input values that produce different outputs is returned,
    otherwise None is returned.

    Note that, in order to check the equivalence, the input and the output dimensions of the two models must be the
    same, otherwise an exception is raised.

    Due to arithmetic errors, it is strongly recommended to provide input bounds to the method, in order to limit the
    impact of such arithmetic errors for very high input values, which could bring to consistent discrepancies in
    the output values. In fact, arithmetic errors and unlimited search space of the counterexample can easily bring
    to a satisfiable formula, even if the two models are strictly equivalent.

    Low values of the epsilon parameter should be combined to tight input bounds to ensure the correctness of the
    equivalence check. For instance, the default epsilon of 1e-6 is recommended for models with input bounds in the
    range [-1, 1] or [0, 1].

    :param model1: The first model
    :param model2: The second model
    :param input_bounds: A list of tuples [(l1, u1), ..., (ln, un)] specifying the inclusive lower and upper bounds
                        for each input variable of the models. If not provided, the bounds are set to (-inf, inf).
    :param epsilon: The tolerance for the equivalence check due to floating-point arithmetic
    :return: a tuple containing a boolean indicating if the two models are strictly equivalent and a counterexample
     if they are not
    """
    return _equivalence_check(model1, model2, input_bounds, are_not_strict_equivalent_formula(epsilon))


def are_not_strict_equivalent_formula(epsilon: float) -> Callable[[list[Real], list[Real]], BoolRef]:
    return lambda out1, out2: Not(And([Abs(out1[i] - out2[i]) < epsilon for i in range(len(out1))]))


def are_approximate_equivalent(model1: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                               model2: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                               input_bounds: List[Tuple[float, float] | List[Tuple[float, float]]] | None = None,
                               p: float = 1,
                               epsilon: float = 1e-6):
    """
    Checks if two feed-forward models are (p, epsilon)-approximately equivalent.
    Given two real values p and epsilon, two models are (p, epsilon)-approximately equivalent if for every possible
    input, the p-norm of the difference between their output vectors is less than epsilon.

    Note that, based on the value of p, the (p, epsilon)-approximate equivalence uses different distance metrics:
    - For p = 1, the (1, epsilon)-approximate equivalence uses the Manhattan distance;
    - For p = 2, the (2, epsilon)-approximate equivalence uses the Euclidean distance;
    - For p = float('inf'), the (inf, epsilon)-approximate equivalence uses the Maximum distance, and hence
        the equivalence check is performed equivalently to the strict equivalence check for the given epsilon.

    The equivalence is checked by proving the unsatisfiability of the formula that represents the negation of the
    nearness between each of the two models output vectors.
    Besides the boolean value representing the equivalence, if the two models are not (p, epsilon)-approximately
    equivalent, a counterexample consisting in a list of input values that produce distant output vectors is returned,
    otherwise None is returned.

    Note that, in order to check the equivalence, the input and the output dimensions of the two models must be the
    same, otherwise an exception is raised.

    :param model1: The first model
    :param model2: The second model
    :param input_bounds: A list of tuples [(l1, u1), ..., (ln, un)] specifying the inclusive lower and upper bounds
                        for each input variable of the models. If not provided, the bounds are set to (-inf, inf).
    :param p: The norm to use for the distance metric
    :param epsilon: The tolerance for the approximate equivalence check
    :return: a tuple containing a boolean indicating if the two models are (p, epsilon)-approximately equivalent and a
     counterexample if they are not
    """
    if p == float('inf'):
        return are_strict_equivalent(model1, model2, input_bounds, epsilon)

    return _equivalence_check(model1, model2, input_bounds, are_not_approximate_equivalent_formula(p, epsilon))


def are_not_approximate_equivalent_formula(p: float, epsilon: float) -> Callable[[list[Real], list[Real]], BoolRef]:
    return lambda out1, out2: simplify(Sum([Abs(out1[i] - out2[i]) ** p for i in range(len(out1))]) ** (1 / p)) >= epsilon


def are_argmax_equivalent(model1: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                          model2: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                          input_bounds: List[Tuple[float, float] | List[Tuple[float, float]]] | None = None):
    """
    Checks if two feed-forward models are argmax equivalent.
    """
    return _equivalence_check(model1, model2, input_bounds, are_not_argmax_equivalent_formula())


def are_not_argmax_equivalent_formula() -> Callable[[list[Real], list[Real]], BoolRef]:
    out_array_1, out_array_2 = Array('out1', IntSort(), RealSort()), Array('out2', IntSort(), RealSort())
    i1, i2 = Int('i1'), Int('i2')
    return lambda out1, out2: And(And([out_array_1[i] == out1[i] for i in range(len(out1))] +
                                        [out_array_2[i] == out2[i] for i in range(len(out2))]),
                                  And([i1 >= 0, i1 < len(out1), i2 >= 0, i2 < len(out1)]),
                                  _argmaxis_formula(out_array_1, i1, len(out1)),
                                  _argmaxis_formula(out_array_2, i2, len(out1)),
                                  i1 != i2)


def _argmaxis_formula(out_array: Array, i: Int, m: int) -> BoolRef:
    """
    Returns the formula that represents the argmaxis function of the given outuput vector, which is true
    if the i-th element of the output vector is the maximum element of the vector.

    :param out_array: The output vector as a Z3 array
    :param i: The index of the element to check
    :param m: The size of the output vector
    """
    return And([Or(And(j < i, out_array[i] > out_array[j]), And(i <= j, out_array[i] >= out_array[j]))
                for j in range(m)])


#Checks if the input and output dimensions of the two models are the same.
#If the dimensions are different, a ValueError is raised.
def _check_models_dimensions(model1: NeuralNetwork, model2: NeuralNetwork):
    if model1.input_size() != model2.input_size():
        raise ValueError(
            f"Expected models with the same input dimension, but got {model1.input_size()} and {model2.input_size()}")
    if model1.output_size() != model2.output_size():
        raise ValueError(
            f"Expected models with the same output dimension, but got {model1.output_size()} and {model2.output_size()}")


def _equivalence_check(model1: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                       model2: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                       input_bounds: List[Tuple[float, float] | List[Tuple[float, float]]],
                       output_formula: Callable[[List[Real], List[Real]], BoolRef]) \
        -> Tuple[bool, List[float] | None]:
    model1, model2 = convert_models_to_neural_networks(model1, model2)

    # Check if the input and output dimensions of the two models are the same
    _check_models_dimensions(model1, model2)

    # Encode the input-output relation of the two models into SMT formulas
    formula1, inputs1, outputs1 = encode_into_SMT_formula(model1, input_bounds=input_bounds, var_prefix="m1_")
    formula2, inputs2, outputs2 = encode_into_SMT_formula(model2, input_bounds=input_bounds, var_prefix="m2_")

    # Build the strict equivalence formula
    inputs_equivalence = And([inputs1[i] == inputs2[i] for i in range(len(inputs1))])
    formula = And(inputs_equivalence, formula1, formula2, output_formula(outputs1, outputs2))

    # Check if the formula is satisfiable (i.e., the two models are not strictly equivalent)
    satisfiability, input_values = get_float_formula_satisfiability(formula, inputs1)
    return not satisfiability, input_values
