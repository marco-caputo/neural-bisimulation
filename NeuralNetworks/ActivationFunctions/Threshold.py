import ActivationFunction
from z3 import BoolRef, If, Real


class Threshold(ActivationFunction):

    def __init__(self, threshold: float = 0.5, value: float = 0):
        super().__init__()
        self.threshold = threshold
        self.value = value

    def apply(self, x: float) -> float:
        return x if x > self.threshold else self.value

    def formula(self, x: Real) -> BoolRef:
        return If(x > self.threshold, x, self.value)
