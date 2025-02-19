from NeuralNetworks.ActivationFunctions import ActivationFunction
from z3 import If, Real, ArithRef


class HardTanh(ActivationFunction):
    """
    HardTanh activation function.
    This function is defined as:
        f(x) = max(**min_val**, min(**max_val**, x))
    """

    def __init__(self, min_val: float = -1, max_val: float = 1):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def apply(self, x: float) -> float:
        if x <= self.min_val:
            return self.min_val
        elif x >= self.max_val:
            return self.max_val
        else:
            return x

    def formula(self, x: Real) -> ArithRef:
        return If(x <= self.min_val, self.min_val,
                  If(x >= self.max_val, self.max_val,
                     x))