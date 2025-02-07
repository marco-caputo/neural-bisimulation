import ActivationFunction
from z3 import BoolRef, If, Real


class Identity(ActivationFunction):

    def __init__(self):
        super().__init__()

    def apply(self, x: float) -> float:
        return x

    def formula(self, x: Real) -> BoolRef:
        return x
