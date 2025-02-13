from NeuralNetworks.ActivationFunctions import ActivationFunction
from z3 import BoolRef, Real


class Identity(ActivationFunction):
    """
    Identity activation function.
    This function is defined as:
        f(x) = x
    """

    def __init__(self):
        super().__init__()

    def apply(self, x: float) -> float:
        return x

    def formula(self, x: Real) -> BoolRef:
        return x
