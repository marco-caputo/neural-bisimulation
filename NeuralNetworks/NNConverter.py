from z3 import Real, Sum, And, BoolRef

from AQTSMetrics import SPA
from ApproxBisimulation import FiniteStateProcess
from SMTEquivalence import get_optimal_solution, is_satisfiable
from NeuralNetworks.NeuralNetwork import NeuralNetwork

ACTION = "a"
START_STATE = "s"


def to_fsp(model: NeuralNetwork,
           input_bounds: list[tuple[float, float]] = None,
           shallow_bounds: bool = False) -> FiniteStateProcess:
    """
    Converts a neural network to a Finite State Process (FSP) model.
    The FSP model is obtained by encoding the neural network as a set of states and transitions between them where:
    - a state is created for each node in each layer of the network, representing that node having the maximum value in
        its layer,
    - a transition is added between two states if, according to weights and input bounds, the state of a node having
        a maximum value in its layer can lead to a node in the next layer having the maximum value in its layer.

    If the shallow_bounds flag is set to True, transitions from layers i-1 to i are added by considering shallow
    constraints on output variables of the previous layers, so just by considering the maximum and minimum values
    that the output variables can assume without any additional constraints from the previous layers.
    This can lead to a more efficient encoding but might not be as precise as the default approach.

    Every action in the produced FSP model is represented by the string "a".

    :param model: a NeuralNetwork model,
    :param input_bounds: A list of tuples [(l1, u1), ..., (ln, un)] specifying the inclusive lower and upper bounds
                          for each input variable.
    :param shallow_bounds: A boolean flag indicating whether to use shallow bounds for the transitions between layers.
    :return: A FiniteStateProcess model representing the given neural network.
    """

    # Declare output variables for each layer including the input layer
    variables_per_layer: list[list[Real]] = [[Real(f'x{i}') for i in range(model.input_size())]]
    for i, layer in enumerate(model.layers, 1):
        variables_per_layer.append([Real(f'h{j}_{i}') for j in range(len(layer))])

    # Encode hidden constraints if any. This consists of just input constraints if shallow_bounds is False
    if input_bounds is not None:
        if len(input_bounds) != model.input_size():
            raise ValueError(f"Expected {model.input_size()} input bounds, but got {len(input_bounds)}")
        hidden_outputs_constraints = _get_hidden_outputs_constraints(model, input_bounds, variables_per_layer, shallow_bounds)
    else:
        hidden_outputs_constraints = [[] for _ in range(len(model.layers) + 1)]

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

    # Add transitions from the start state to the input variables
    for input_variable in variables_per_layer[0]:
        fsp.add_transition(START_STATE, ACTION, str(input_variable))

    affine_expressions = []

    # Add transitions for those couples of nodes that satisfy the constraints.
    for i, variables in enumerate(variables_per_layer[1:], 1):

        if shallow_bounds:
            affine_expressions.clear()

        # Set the expression for the affine transformation from the whole previous layer to the current layer
        z_variables = [Real(f'z{j}_{i}') for j in range(len(variables))]
        affine_expressions.append(And(
            [variables_per_layer[i][j] == model.layers[i-1].activation_functions[j].formula(z_variables[j])
             for j in range(len(variables))] +
            [z_variables[j] == (Sum([variables_per_layer[i - 1][k] * model.layers[i - 1].weights[k][j]
                   for k in range(model.layers[i - 1].input_size())]) + model.layers[i - 1].biases[j])
             for j in range(len(variables))] +
            hidden_outputs_constraints[i - 1]
        ))

        # Check if the constraints are satisfiable and add the transition if so
        for j, v in enumerate(variables):
            for k, w in enumerate(variables_per_layer[i - 1]):
                if is_satisfiable(And(And(affine_expressions),
                                      is_max_constraints[i][j],
                                      is_max_constraints[i - 1][k])):
                    fsp.add_transition(str(w), ACTION, str(v))

    return fsp


def to_spa(model: NeuralNetwork,
           input_bounds: list[tuple[float, float]] = None,
           shallow_bounds: bool = False) -> SPA:
    """
    Converts a neural network to a SPA model.
    The SPA model is obtained by encoding the neural network as a set of states and transitions between them where:
    - a state is created for each node in each layer of the network, representing the state where that node has the
        maximum value in its layer,
    - a probabilistic transition is added between two states if, according to weights and input bounds, the state of
        a node having a maximum value in its layer can lead to a node in the next layer having the maximum value in
        its layer.

    If the shallow_bounds flag is set to True, transitions from layers i-1 to i are added by considering shallow
    constraints on output variables of the previous layers, so just by considering the maximum and minimum values
    that the output variables can assume without any additional constraints from the previous layers.
    This can lead to a more efficient encoding but might not be as precise as the default approach.

    The probability distribution is simply defined setting the possible transitions to have equal probabilities.

    :param model: a NeuralNetwork model,
    :param input_bounds: A list of tuples [(l1, u1), ..., (ln, un)] specifying the inclusive lower and upper bounds
                          for each input variable.
    :param shallow_bounds: A boolean flag indicating whether to use shallow bounds for the transitions between layers.
    :return: A SPA model representing the given neural network.
    """
    return to_fsp(model, input_bounds, shallow_bounds).to_spa()


def _get_hidden_outputs_constraints(model: NeuralNetwork,
                                    input_bounds: list[tuple[float, float]],
                                    variables_per_layer: list[list[Real]],
                                    shallow_bounds: bool) \
        -> list[list[BoolRef]]:
    """
    Returns a list of constraints for the hidden layer outputs of the given model, including the output layer.
    These constraint define the maximum and the minimum values the intermediate hidden outputs of the network can assume
    and are obtained through optimization problems defined by the weights in the given model and input bounds.

    :param model: a NeuralNetwork model,
    :param input_bounds: A list of tuples [(l1, u1), ..., (ln, un)] specifying the inclusive lower and upper bounds
                          for each input variable.
    :param variables_per_layer: A list of lists of Z3 variables representing the output variables of each layer.
    :param shallow_bounds: A boolean flag indicating whether to use shallow bounds for the transitions between layers.
    """

    # List of lists of constraints to be returned indicating lower and upper bounds for each variable in variables_per_layer
    constraints = [[]]
    if len(input_bounds) != model.input_size():
        raise ValueError(f"Expected {model.input_size()} input bounds, but got {len(input_bounds)}")

    # List used to store bound constraints for variables in the previous layer
    previous_variables_and_bounds = [(variables_per_layer[0][i], l, u) for i, (l, u) in enumerate(input_bounds)]
    _add_constraints_to(previous_variables_and_bounds, constraints[-1])

    if not shallow_bounds:
        constraints = constraints + [[] for _ in range(len(model.layers))]
        return constraints

    # List used to store constraints incrementally until the current layer
    current_constraints = []

    for i, layer in enumerate(model.layers, 1):

        # List used to store constraints for variables in the current layer in form of tuples (variable, lower_bound, upper_bound)
        current_variables_and_bounds = []

        for j in range(len(layer)):
            _add_constraints_to(previous_variables_and_bounds, current_constraints)

            z = Real(f'z{j}_{i}')
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
