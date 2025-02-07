import ActivationFunction
from z3 import BoolRef, If, Real
class HardSigmoid(ActivationFunction):

    def __init__(self):
        super().__init__()

    def apply(self, x: float) -> float:
        if x <= -3:
            return 0
        elif x >= 3:
            return 1
        else:
            return x / 6 + 0.5

    def formula(self, x: Real) -> BoolRef:
        return If(x <= -3, 0,
                  If(x >= 3, 1,
                     x / 6 + 0.5))