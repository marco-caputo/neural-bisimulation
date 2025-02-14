from z3 import Real, Sum, And, BoolRef

from AQTSMetrics import SPA
from ApproxBisimulation import FiniteStateProcess
from SMTEquivalence.SMTUtils import get_optimal_solution, is_satisfiable
from NeuralNetworks.NeuralNetwork import NeuralNetwork

ACTION = "a"
START_STATE = "s"


def to_fsp(model: NeuralNetwork, input_bounds: list[tuple[float, float]] = None) -> FiniteStateProcess:
    """
    Converts a neural network to a Finite State Process (FSP) model.
    The FSP model is obtained by encoding the neural network as a set of states and transitions between them where:
    - a state is created for each node in each layer of the network, representing that node having the maximum value in
        its layer,
    - a transition is added between two states if, according to weights and input bounds, the state of a node having
        a maximum value in its layer can lead to a node in the next layer having the maximum value in its layer.

    Every action in the produced FSP model is represented by the string "a".

    :param model: a NeuralNetwork model,
    :param input_bounds: A list of tuples [(l1, u1), ..., (ln, un)] specifying the inclusive lower and upper bounds
                          for each input variable.
    :return: A FiniteStateProcess model representing the given neural network.
    """

    # Declare output variables for each layer including the input layer
    variables_per_layer: list[list[Real]] = [[Real(f'x{i}') for i in range(model.input_size())]]
    for i, layer in enumerate(model.layers, 1):
        variables_per_layer.append([Real(f'h{i}_{j}') for j in range(len(layer))])

    # Encode hidden constraints if any
    hidden_outputs_constraints = [[] for _ in range(len(model.layers) + 1)]
    if input_bounds is not None:
        if len(input_bounds) != model.input_size():
            raise ValueError(f"Expected {model.input_size()} input bounds, but got {len(input_bounds)}")
        hidden_outputs_constraints = _get_hidden_outputs_constraints(model, input_bounds, variables_per_layer)

    # Initialize the FSP model
    states = {START_STATE}
    for variables in variables_per_layer:
        states.update([str(v) for v in variables])
    fsp = FiniteStateProcess(states, "s")

    # Define max constraints for each node in each layer
    is_max_constraints = [[None for _ in range(len(variables))] for variables in variables_per_layer]
    for i, variables in enumerate(variables_per_layer):
        for j in range(len(variables)):
            is_max_constraints[i][j] = And([variables[j] > variables[k] for k in range(len(variables)) if k != j])

    # Add transitions for those couples of nodes that satisfy the constraints.
    for i, variables in enumerate(variables_per_layer[1:], 1):
        for j, v in enumerate(variables):
            affine_expr = variables_per_layer[i][j] == (
                    Sum([variables_per_layer[i - 1][k] * model.layers[i - 1].weights[k][j]
                         for k in range(model.layers[i - 1].input_size())])
                    + model.layers[i - 1].biases[j]
            )
            for k, w in enumerate(variables_per_layer[i - 1]):
                if is_satisfiable(And(affine_expr,
                                      is_max_constraints[i][j],
                                      is_max_constraints[i - 1][k],
                                      And(hidden_outputs_constraints[i - 1]))):
                    fsp.add_transition(str(w), ACTION, str(v))

    return fsp


def to_spa(model: NeuralNetwork, input_bounds: list[tuple[float, float]] = None) -> SPA:
    """
    Converts a neural network to a SPA model.
    The SPA model is obtained by encoding the neural network as a set of states and transitions between them where:
    - a state is created for each node in each layer of the network, representing the state where that node has the
        maximum value in its layer,
    - a probabilistic transition is added between two states if, according to weights and input bounds, the state of
        a node having a maximum value in its layer can lead to a node in the next layer having the maximum value in
        its layer.

    The probability distribution is simply defined setting the possible transitions to have equal probabilities.

    :param model: a NeuralNetwork model,
    :param input_bounds: A list of tuples [(l1, u1), ..., (ln, un)] specifying the inclusive lower and upper bounds
                          for each input variable.
    :return: A SPA model representing the given neural network.
    """
    return to_fsp(model, input_bounds).to_spa()


def _get_hidden_outputs_constraints(model: NeuralNetwork,
                                    input_bounds: list[tuple[float, float]],
                                    variables_per_layer: list[list[Real]]) \
        -> list[list[BoolRef]]:
    """
    Returns a list of constraints for the hidden layer outputs of the given model, including the output layer.
    These constraint define the maximum and the minimum values the intermediate hidden outputs of the network can assume
    and are obtained through optimization problems defined by the weights in the given model and input bounds.

    :param model: a NeuralNetwork model,
    :param input_bounds: A list of tuples [(l1, u1), ..., (ln, un)] specifying the inclusive lower and upper bounds
                          for each input variable.
    :param variables_per_layer: A list of lists of Z3 variables representing the output variables of each layer.
    """

    # List of lists of constraints to be returned indicating lower and upper bounds for each variable in variables_per_layer
    constraints = [[]]
    if len(input_bounds) != model.input_size():
        raise ValueError(f"Expected {model.input_size()} input bounds, but got {len(input_bounds)}")

    # List used to store bound constraints for variables in the previous layer
    previous_variables_and_bounds = [(variables_per_layer[0][i], l, u) for i, (l, u) in enumerate(input_bounds)]
    _add_constraints_to(previous_variables_and_bounds, constraints[-1])

    for i, layer in enumerate(model.layers, 1):

        # List used to store constraints for variables in the current layer in form of tuples (variable, lower_bound, upper_bound)
        current_variables_and_bounds = []

        for j in range(len(layer)):
            # List used to store constraints for the current optimization problem
            current_constraints = []
            _add_constraints_to(previous_variables_and_bounds, current_constraints)

            z = Real(f'z{i}_{j}')
            affine_expr = Sum(
                [variables_per_layer[i - 1][k] * layer.weights[k][j] for k in range(layer.input_size())]) + \
                          layer.biases[j]
            current_constraints.append(z == affine_expr)
            current_constraints.append(variables_per_layer[i][j] == layer.activation_functions[j].formula(z))

            _, l = get_optimal_solution(variables_per_layer[i][j], current_constraints, maximize=False)
            _, u = get_optimal_solution(variables_per_layer[i][j], current_constraints, maximize=True)

            current_variables_and_bounds.append((variables_per_layer[i][j], l, u))

        constraints.append(list())
        _add_constraints_to(current_variables_and_bounds, constraints[-1])
        previous_variables_and_bounds = current_variables_and_bounds

    return constraints


# Adds upper and lower bounds constraints in form of tuples (variable, lower_bound, upper_bound) to the given list of constraints
def _add_constraints_to(current_variables_and_bounds: list[tuple[Real, float, float]], constraints: list):
    for t in current_variables_and_bounds:
        if t[1] is not None and t[1] != float('-inf'):
            constraints.append(t[0] >= t[1])
        if t[2] is not None and t[2] != float('inf'):
            constraints.append(t[0] <= t[2])
