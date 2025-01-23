from typing import List, Tuple

from z3 import *

def get_float_formula_satisfiability(formula: BoolRef, inputs: List[float]) -> Tuple[bool, List[float] | None]:
    """
    Checks the satisfiability of a Z3 formula involving the given list of real variables as inputs and returns a
    counterexample if the formula is satisfiable.

    :param formula: A Z3 formula
    :param inputs: A list of Z3 real variables involved in the formula
    :return: A tuple containing a boolean indicating if the formula is satisfiable and a counterexample if it is
    """
    s = Solver()
    s.add(formula)
    result = s.check()
    if result == sat:
        model = s.model()
        return True, [float(model.evaluate(x).as_decimal(10).rstrip('?')) for x in inputs]
    elif result == unsat:
        return False, None
    else:
        raise RuntimeError("Solver returned 'unknown'. The equivalence might be too complex to decide.")