import ActivationFunction
from z3 import BoolRef, If, Real
class ReLU(ActivationFunction):

    def __init__(self, max_val: float = float('inf'), threshold: float = 0.0, negative_slope: float = 0.0):
        super().__init__()
        self.max_val = max_val
        self.threshold = threshold
        self.negative_slope = negative_slope

    def apply(self, x: float) -> float:
        if x >= self.max_val:
            return self.max_val
        elif x >= self.threshold:
            return x
        else:
            return self.negative_slope * (x - self.threshold)

    def formula(self, x: Real) -> BoolRef:
        formula = If(x >= self.threshold, x, self.negative_slope * (x - self.threshold))
        if self.max_val != float('inf'): formula = If(x >= self.max_val, self.max_val, formula)
        return formula