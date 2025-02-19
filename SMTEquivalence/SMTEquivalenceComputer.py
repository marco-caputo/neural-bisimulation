import math
from typing import List, Tuple
import torch
import tensorflow as tf
from z3 import *

from NeuralNetworks import NeuralNetwork
from SMTEquivalence import (encode_into_SMT_formula,
                            are_not_approximate_equivalent_formula,
                            get_float_formula_satisfiability)


def compute_approximate_equivalence(model1: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                                    model2: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                                    input_bounds: list[tuple[float, float]] | None = None,
                                    p: float = 1, precision: float = 0.01,
                                    lower: float = 0, upper: float = 1,
                                    verbose: bool = False) -> tuple[float, tuple[float, float]]:
    """
        Computes the approximate equivalence metric between two neural networks using the p-norm distance.
        The metric is defined as the lowest value of difference epsilon such that the two networks are
        epsilon-close in the p-norm distance. The closer the metric to 0, the more equivalent the networks are.
        This value of the metric is approximated using binary search with the given precision. So the first
        returned value corresponds to the approximate value of the metric with an error of at most precision, while
        the second value is a tuple containing the lower and upper bounds were the true value of the metric
        falls.

        Input bounds should be a list of tuples containing, respectively:
        - (-2σ, 2σ) for real-valued inputs with standard deviation σ;
        - [(0 ,0), (1, 1)] for binary-valued inputs.
        - [(0, 0), (1, 1), ..., (n, n)] for inputs with n possible values ranging from 0 to n.

        Lower and upper parameters indicate the initial interval for the binary search, with upper corresponding
        to the maximum possible value of the metric.
        For instance, in case of neural networks having a single output neuron with a sigmoid activation function, the
        equivalence metric is comprised between 0 and 1, so lower and upper should be set to 0 and 1 respectively
        (this is the default value).

        :param model1: The first neural network model.
        :param model2: The second neural network model.
        :param input_bounds: The bounds of the input values for the neural networks. (Defaults to None)
        :param p: The p-norm distance to be used for the metric. (Defaults to 1)
        :param precision: The precision of the binary search. (Defaults to 0.01)
        :param lower: The lower bound of the metric. (Defaults to 0)
        :param upper: The upper bound of the metric. (Defaults to 1)
        :param verbose: Whether to print the results of each iteration of the binary search. (Defaults to False)
    """
    # Encode the input-output relation of the two models into SMT formulas
    formula1, inputs1, outputs1 = encode_into_SMT_formula(model1, input_bounds=input_bounds, var_prefix="m1_")
    formula2, inputs2, outputs2 = encode_into_SMT_formula(model2, input_bounds=input_bounds, var_prefix="m2_")
    constraints = [And([inputs1[i] == inputs2[i] for i in range(len(inputs1))]), formula1, formula2]

    for i in range(math.ceil(math.log(1 / precision, 2))):
        epsilon = (lower + upper) / 2
        constraints.append(are_not_approximate_equivalent_formula(p, epsilon)(outputs1, outputs2))
        satisfiable, counterexample = get_float_formula_satisfiability(And(constraints), inputs1)
        if satisfiable:  # Not (p,epsilon)-approximate equivalent
            lower = epsilon
        else:
            upper = epsilon

        if verbose:
            print(_verbose_string(i, not satisfiable, epsilon, counterexample))

        constraints.pop()

    return (lower + upper) / 2, (lower, upper)


def compute_approximate_equivalence_experiment(model1: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                                    model2: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                                    input_bounds: list[tuple[float, float]] | None = None,
                                    p: float = 1, precision: float = 0.01,
                                    lower: float = 0, upper: float = 1,
                                    verbose: bool = False) -> tuple[float, tuple[float, float], list[list[float]]]:
    # Encode the input-output relation of the two models into SMT formulas
    formula1, inputs1, outputs1 = encode_into_SMT_formula(model1, input_bounds=input_bounds, var_prefix="m1_")
    formula2, inputs2, outputs2 = encode_into_SMT_formula(model2, input_bounds=input_bounds, var_prefix="m2_")
    constraints = [And([inputs1[i] == inputs2[i] for i in range(len(inputs1))]), formula1, formula2]
    counterexamples = []

    for i in range(math.ceil(math.log(1 / precision, 2))):
        epsilon = (lower + upper) / 2
        constraints.append(are_not_approximate_equivalent_formula(p, epsilon)(outputs1, outputs2))
        satisfiable, counterexample = get_float_formula_satisfiability(And(constraints), inputs1)
        if satisfiable:  # Not (p,epsilon)-approximate equivalent
            lower = epsilon
        else:
            upper = epsilon

        if verbose:
            print(_verbose_string(i, not satisfiable, epsilon, counterexample))

        counterexamples.append(counterexample)
        constraints.pop()

    return (lower + upper) / 2, (lower, upper), counterexamples

"""
def compute_approximate_equivalence_2(model1: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                                    model2: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                                    input_bounds: List[Tuple[float, float] | List[Tuple[float, float]]] | None = None,
                                    p: float = 1, precision: float = 0.01,
                                    lower: float = 0, upper: float = 1,
                                    verbose: bool = False) -> tuple[float, tuple[float, float]]:
    for i in range(math.ceil(math.log(1 / precision, 2))):
        epsilon = (lower + upper) / 2
        are_equivalent, counterexample = are_approximate_equivalent(model1, model2, input_bounds, p, epsilon)

        if are_equivalent:
            upper = epsilon
        else:
            lower = epsilon

        if verbose:
            print(_verbose_string(i, are_equivalent, epsilon, counterexample))

    return (lower + upper) / 2, (lower, upper)
"""


def _verbose_string(i: int, are_equivalent: bool, epsilon: float, counterexample: List[float] | None) -> str:
    return f"Iteration {i+1}: " + \
           ("No counterexample" if are_equivalent else f"Found counterexample {counterexample}") + \
           f" for epsilon = {epsilon}"




