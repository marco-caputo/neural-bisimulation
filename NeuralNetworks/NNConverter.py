from typing import Iterable

from werkzeug.datastructures import MultiDict
from z3 import Real, Sum, And, BoolRef
import numpy as np
from scipy.stats import truncnorm

from AQTSMetrics import SPA, DeterministicSPA
from ApproxBisimulation.PFSP import ProbabilisticFiniteStateProcess

from SMTEquivalence import get_optimal_solution, is_satisfiable
from NeuralNetworks.NeuralNetwork import NeuralNetwork

ACTION = "a"
START_STATE = "s"


def to_fsp(model: NeuralNetwork,
           input_bounds: list[tuple[float, float]] = None,
           shallow_bounds: bool = False) -> ProbabilisticFiniteStateProcess:
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
    :return: A ProbabilisticFiniteStateProcess model representing the given neural network.
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
    fsp = ProbabilisticFiniteStateProcess(states, START_STATE)

    # Define max constraints for each node in each layer
    is_max_constraints = [[None for _ in range(len(variables))] for variables in variables_per_layer]
    for i, variables in enumerate(variables_per_layer):
        for j in range(len(variables)):
            is_max_constraints[i][j] = And([variables[j] > variables[k] for k in range(len(variables)) if k != j])

    # Add transitions from the start state to the input variables
    fsp.add_distribution(START_STATE, ACTION, {
        str(input_variable): 1 / len(variables_per_layer[0]) for input_variable in variables_per_layer[0]
    })

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

        transitions = dict()

        # Check if the constraints are satisfiable and add the transition if so
        for j, v in enumerate(variables):
            for k, w in enumerate(variables_per_layer[i - 1]):
                if is_satisfiable(And(And(affine_expressions),
                                      is_max_constraints[i][j],
                                      is_max_constraints[i - 1][k])):
                    if str(w) not in transitions:
                        transitions[str(w)] = []
                    transitions[str(w)].append(str(v))

        for w, vs in transitions.items():
            fsp.add_distribution(str(w), ACTION, {str(v): 1 / len(vs) for v in vs})

    return fsp


def to_spa_smt(model: NeuralNetwork,
           input_bounds: list[tuple[float, float]] = None,
           shallow_bounds: bool = False) -> SPA:
    """
    Converts a neural network to a DSPA model.
    The DSPA model is obtained by encoding the neural network as a set of states and transitions between them where:
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


def to_spa_probabilistic(model: NeuralNetwork,
                        input_bounds: list[tuple[float, float] | list[tuple[float, float]]] | None = None,
                        number_of_samples: int = 10000,
                        mean: float = 0.0, std_deviation: float = 1.0,
                        seed: int = None) -> DeterministicSPA:
    """
    Converts a neural network to a DSPA model.
    The DSPA model is obtained by encoding the neural network as a set of states and transitions between them where:
    - a state is created for each node in each layer of the network, representing the state where that node has the
        maximum value in its layer,
    - a probabilistic transition is added between two states if, according to weights and input bounds, the state of
        a node having a maximum value in its layer can lead to a node in the next layer having the maximum value in
        its layer.

    The probability of each transition is derived heuristically by observing the hidden outputs of the network from
    random input samples, obtained by a normal distribution with the given input bounds, mean and standard deviation.
    The default values for the mean and standard deviation are 0 and 1, respectively, which correspond to the typical
    values used for data scaling in machine learning.
    In-scale lower and upper bounds can be specified to limit the range of the generated samples for each input feature.
    For those input bounds having multiple intervals or describing a discrete distribution (multiple intervals having a
    lower bound equal to the upper bound) the samples are generated uniformly from the given intervals.

    :param model: a NeuralNetwork model.
    :param input_bounds: A list of tuples [(l1, u1), ..., (ln, un)] or a list of lists of tuples
    [[(l1, u1), ..., (ln, un)], ..., [(l1, u1), ..., (ln, un)]] specifying the inclusive lower and upper bounds.
    :param number_of_samples: The number of samples to draw from the normal distribution.
    :param mean: The mean of the normal distribution.
    :param std_deviation: The standard deviation of the normal distribution.
    :param seed: The seed for the random sample generator.
    """
    return DeterministicSPA(_to_spa_data(model, input_bounds, number_of_samples, mean, std_deviation, seed))


def to_pfsp_probabilistic(model: NeuralNetwork,
                          input_bounds: list[tuple[float, float] | list[tuple[float, float]]] | None = None,
                          number_of_samples: int = 10000,
                          mean: float = 0.0, std_deviation: float = 1.0,
                          seed: int = None) -> ProbabilisticFiniteStateProcess:
    """
    Converts a neural network to a PFSP model.
    The PFSP model is obtained by encoding the neural network as a set of states and transitions between them where:
    - a state is created for each node in each layer of the network, representing the state where that node has the
        maximum value in its layer,
    - a single action is defined for each transition between states.
    - a single distribution of probabilistic transitions is defined for each state-action pair.
    - a probabilistic transition is added between two states if, according to weights and input bounds, the state of
        a node having a maximum value in its layer can lead to a node in the next layer having the maximum value in
        its layer.

    The probability of each transition is derived heuristically by observing the hidden outputs of the network from
    random input samples, obtained by a normal distribution with the given mean and standard deviation. The default
    values for the mean and standard deviation are 0 and 1, respectively, which correspond to the typical values used
    for data scaling in machine learning.
    In-scale lower and upper bounds can be specified to limit the range of the generated samples.

    :param model: a NeuralNetwork model,
    :param input_bounds: A list of tuples [(l1, u1), ..., (ln, un)] or a list of lists of tuples
    [[(l1, u1), ..., (ln, un)], ..., [(l1, u1), ..., (ln, un)]] specifying the inclusive lower and upper bounds.
    :param number_of_samples: The number of samples to draw from the normal distribution.
    :param mean: The mean of the normal distribution.
    :param std_deviation: The standard deviation of the normal distribution.
    :param seed: The seed for the random sample generator.
    """
    data = _to_spa_data(model, input_bounds, number_of_samples, mean, std_deviation, seed)

    return ProbabilisticFiniteStateProcess(set(data.keys()), START_STATE, {
        s: MultiDict([(ACTION, data[s][ACTION])]) for s in data
    })



def _to_spa_data(model: NeuralNetwork,
                input_bounds: list[tuple[float, float] | list[tuple[float, float]]] | None = None,
                number_of_samples: int = 10000,
                mean: float = 0.0, std_deviation: float = 1.0,
                seed: int = None) -> dict[str, dict[str, dict[str, float]]]:

    if input_bounds is not None and len(input_bounds) != model.input_size():
        raise ValueError(f"Expected {model.input_size()} input bounds, but got {len(input_bounds)}")

    # Declare state strings for each layer including the input layer
    states_per_layer: list[list[str]] = [[f'x{i}' for i in range(model.input_size())]]
    for i, layer in enumerate(model.layers, 1):
        states_per_layer.append([f'h{j}_{i}' for j in range(len(layer))])
    states = [item for sublist in states_per_layer for item in sublist]

    # Initialize the transition counts
    transition_counts = {s: dict() for s in states}
    for i in range(1, len(states_per_layer)):
        for s1 in states_per_layer[i - 1]:
            for s2 in states_per_layer[i]:
                transition_counts[s1][s2] = 0

    # Generate random input samples and count the transitions
    for inputs in _random_vectors(number_of_samples, model.input_size(), input_bounds, mean, std_deviation, seed):
        for i, layer in enumerate(model.layers, 1):
            outputs = layer.forward_pass(inputs)
            s1 = states_per_layer[i - 1][inputs.index(max(inputs))]
            s2 = states_per_layer[i][outputs.index(max(outputs))]
            transition_counts[s1][s2] += 1
            inputs = outputs

    # Normalize the transition counts to obtain the probabilities
    spa_data = {s: {ACTION: dict()} for s in states}
    for s1 in states:
        total = sum(transition_counts[s1].values())
        for s2 in states:
            if s2 in transition_counts[s1] and transition_counts[s1][s2] != 0:
                spa_data[s1][ACTION][s2] = transition_counts[s1][s2] / total

    # Add initial state and transitions
    spa_data[START_STATE] = {ACTION: {s: 1 / len(states_per_layer[0]) for s in states_per_layer[0]}}

    return spa_data


def _random_vectors(num_vectors: int, vector_dim: int,
                    input_bounds: list[tuple[float, float] | list[tuple[float, float]]] | None = None,
                    mean: float = 0., std_dev: float = 1.,
                    seed: int = None) -> list[list[float]]:
    """
    Generates a sequence of random vectors from a Gaussian distribution with the given mean and standard deviation.

    :param num_vectors: Number of random vectors to generate
    :param vector_dim: Dimension of each random vector
    :param input_bounds: A list of tuples [(l1, u1), ..., (ln, un)] or a list of lists of tuples for truncation
    :param mean: Mean of the Gaussian distribution
    :param std_dev: Standard deviation of the Gaussian distribution
    :return: List of random vectors (numpy arrays)
    """

    def _to_list(bounds: tuple) -> list[float]:
        l = [bounds[0] if bounds[0] not in {None, float('-inf')} else -np.inf,
             bounds[1] if bounds[1] not in {None, float('inf')} else np.inf]
        return [(l[0] - mean) / std_dev, (l[1] - mean) / std_dev]

    if input_bounds is None:
        bounds_list = [[-np.inf, np.inf] for _ in range(vector_dim)]
    else:
        bounds_list = []
        for bound in input_bounds:
            if isinstance(bound, tuple):
                bounds_list.append(_to_list(bound))
            else:
                bounds_list.append([_to_list(b) for b in bound])

    rng = np.random.default_rng(seed)  # Create a random generator with seed

    vectors = np.empty((num_vectors, vector_dim))  # Preallocate array
    for i in range(vector_dim):
        if not isinstance(bounds_list[i][0], list):
            vectors[:, i] = truncnorm.rvs(bounds_list[i][0], bounds_list[i][1],
                                          loc=mean, scale=std_dev, size=num_vectors, random_state=rng)
        else:
            for j in range(num_vectors):
                lower_upper = rng.choice(bounds_list[i])
                vectors[j, i] = rng.uniform(lower_upper[0], lower_upper[1])

    return vectors.tolist()


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
