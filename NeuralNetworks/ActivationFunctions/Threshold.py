from NeuralNetworks.ActivationFunctions import ActivationFunction
from z3 import If, Real, ArithRef


class Threshold(ActivationFunction):
    """
    Threshold activation function.
    This function is defined as:
        f(x) = x if x > **threshold**,
            **value** otherwise
    """

    def __init__(self, threshold: float = 0.5, value: float = 0):
        super().__init__()
        self.threshold = threshold
        self.value = value

    def apply(self, x: float) -> float:
        return x if x > self.threshold else self.value

    def formula(self, x: Real) -> ArithRef:
        return If(x > self.threshold, x, self.value)
