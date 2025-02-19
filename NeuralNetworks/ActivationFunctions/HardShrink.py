from NeuralNetworks.ActivationFunctions import ActivationFunction
from z3 import If, Or, Real, ArithRef


class HardShrink(ActivationFunction):
    """
    Hard Shrink activation function.
    This function is defined as:
        f(x) = x if x > **lambd** or x < -**lambd**,
            0 otherwise
    """

    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd

    def apply(self, x: float) -> float:
        return x if x > self.lambd or x < -self.lambd else 0

    def formula(self, x: Real) -> ArithRef:
        return If(Or(x > self.lambd, x < -self.lambd), x, 0)
