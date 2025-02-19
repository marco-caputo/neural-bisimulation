from NeuralNetworks.ActivationFunctions import ActivationFunction
from z3 import Real, And, If, ArithRef, RealVal
import math
import numpy as np


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    This function is defined as:
        f(x) = 1 / (1 + exp(-x))
    """

    #  List of x values for the formula approximation of the sigmoid function
    SAMPLES = [np.log(y / (1 - y)) for y in np.arange(0.005, 1.00, 0.005)]

    def __init__(self):
        super().__init__()

    def apply(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def formula(self, x: Real) -> ArithRef:
        """
        The formula approximation of the sigmoid function is:
        f(x) = 0 if x < f^-1(0.005)
        f(x) = 0.01 if f^-1(0.005) <= x < f^-1(0.015)
        ...
        f(x) = 0.99 if f^-1(0.985) <= x < f^-1(0.995)
        f(x) = 1
        """
        formula = RealVal(0)
        for i in range(1, len(self.SAMPLES)):
            formula = If(And(RealVal(self.SAMPLES[i - 1]) <= x, x < RealVal(self.SAMPLES[i])), RealVal(i) / 100, formula)
        return If(x >= RealVal(self.SAMPLES[-1]), RealVal(1), formula)

