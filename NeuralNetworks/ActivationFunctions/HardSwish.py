from NeuralNetworks.ActivationFunctions import ActivationFunction
from z3 import BoolRef, If, Real

class HardSwish(ActivationFunction):
    """
    Hard Swish activation function, also known as HardSiLU.
    This function is defined as:
        f(x) = x * max(0, min(1, x / 6 + 0.5))
    """

    def __init__(self):
        super().__init__()

    def apply(self, x: float, min_val: float = -1, max_val: float = 1) -> float:
        if x <= -3:
            return 0
        elif x >= 3:
            return x
        else:
            return x * (x / 6 + 0.5)

    def formula(self, x: Real) -> BoolRef:
        return x * \
            If(x <= -3, 0,
               If(x >= 3, 1,
                  x / 6 + 0.5))