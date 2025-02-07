import ActivationFunction
from z3 import BoolRef, If, Or, Real


class HardShrink(ActivationFunction):

    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd

    def apply(self, x: float) -> float:
        return x if x > self.lambd or x < -self.lambd else 0

    def formula(self, x: Real) -> BoolRef:
        return If(Or(x > self.lambd, x < -self.lambd), x, 0)
