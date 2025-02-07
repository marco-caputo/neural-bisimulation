import ActivationFunction
from z3 import BoolRef, If, Real

class HardSwish(ActivationFunction):

    def __init__(self):
        super().__init__()

    def apply(self, x: float, min_val: float = -1, max_val: float = 1) -> float:
        if x <= -3:
            return 0
        elif x >= 3:
            return x
        else:
            return x / 6 + 0.5

    def formula(self, x: Real) -> BoolRef:
        return x * \
            If(x <= -3, 0,
               If(x >= 3, 1,
                  x / 6 + 0.5))