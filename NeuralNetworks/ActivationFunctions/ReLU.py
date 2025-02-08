from NeuralNetworks.ActivationFunctions import ActivationFunction
from z3 import BoolRef, If, Real


class ReLU(ActivationFunction):
    """
    Rectified Linear Unit (ReLU) activation function.
    This function encompasses the Leaky ReLU and the ReLU activation functions, and is defined as:
        f(x) = min(**max_val**, x) if x >= **threshold**,
            **negative_slope** * (x - **threshold**) otherwise
    """

    def __init__(self, max_val: float = None, threshold: float = 0.0, negative_slope: float = 0.0):
        super().__init__()
        self.max_val = max_val
        self.threshold = threshold
        self.negative_slope = negative_slope

    def apply(self, x: float) -> float:
        if self.max_val is not None and x >= self.max_val:
            return self.max_val
        elif x >= self.threshold:
            return x
        else:
            return self.negative_slope * (x - self.threshold)

    def formula(self, x: Real) -> BoolRef:
        formula = If(x >= self.threshold, x, self.negative_slope * (x - self.threshold))
        if self.max_val is not None: formula = If(x >= self.max_val, self.max_val, formula)
        return formula
