from z3 import *
from AQTSMetrics import DeterministicSPA
from SMTEquivalence import get_optimization_problem_formula


def hd_formula(spa: DeterministicSPA, d: Array, tau: dict[str, Real], tau1: dict[str, Real], var_suffix: str) \
        -> tuple[BoolRef, Real]:
    """
    Returns the Z3 formula for the hd optimization problem and the variable y that represents the optimal
    value of the objective function.
    """
    x_array = Array(f'x_array_{var_suffix}', IntSort(), RealSort())
    objective = Sum([(tau.get(s, 0.) - tau1.get(s, 0.)) * x_array[spa.index_of(s)] for s in spa.states])
    constraints = ([And(0 <= x_array[spa.index_of(s)], x_array[spa.index_of(s)] <= 1) for s in spa.states] +
                   [x_array[spa.index_of(s)] - x_array[spa.index_of(t)] <= d[spa.index_of(s)][spa.index_of(t)]
                    for s in spa.states for t in spa.states])
    constraints = simplify(And(constraints))

    return get_optimization_problem_formula(objective, constraints, [x_array], maximize=True, suffix=var_suffix)


def haus_formula(spa: DeterministicSPA, d: Array, a: str, s: str, t: str) -> tuple[BoolRef, BoolRef]:
    """
    Returns the couple of Z3 formulas for the haus optimization problem that respectively represent the existence of
    a feasible solution and the unboundedness of the problem.
    """
    if spa.distribution(s, a) is None or spa.distribution(t, a) is None:
        raise ValueError(f'Action {a} not found in the state {s} or {t}.')

    hd_formula_1, hd_y_1 = hd_formula(spa, d, spa.distribution(s, a), spa.distribution(t, a), f'{s}_{t}_1')
    hd_formula_2, hd_y_2 = hd_formula(spa, d, spa.distribution(t, a), spa.distribution(s, a), f'{s}_{t}_2')
    constraints = [hd_y_1 == hd_y_2, hd_formula_1, hd_formula_2]

    not_empty_formula = simplify(And(constraints))                   # A solution exists
    unbounded_formula = And(not_empty_formula, hd_y_1 > 1e10)  # Unbounded solution

    return not_empty_formula, unbounded_formula


def postfixpoint_1(spa: DeterministicSPA, d: Array, s: str, t: str) -> tuple[BoolRef, BoolRef]:
    """
    Verifies if the postfixpoint_1 property holds for the given deterministic SPA model, states and current value of m.

    The postfixpoint_1 property is checked by taking into consideration different optimization problems for each action
    that can be executed in the states s and t.
    If the maximum value of the objective function is unbounded for at least one action, the whole problem is unbounded,
    so the property does not hold. Otherwise, if there exists at least an optimal solution for an action, the problem
    is bounded, and the property holds.
    """
    if spa.actions(s) != spa.actions(t):
        raise ValueError(f'The necessary condition for postfixpoint1 is not satisfied '
                         f'(states {s} and {t} are not defined on the same set of actions).')

    # Checks if there is an optimal solution among all actions
    not_empty_constraints = []  # Solution exists
    unbounded_constraints = []  # Solution exists and it is unbounded
    for a in spa.actions(s):
        not_empty_formula, unbounded_formula = haus_formula(spa, d, a, s, t)
        not_empty_constraints.append(not_empty_formula)
        unbounded_constraints.append(unbounded_formula)

    return simplify(Or(not_empty_constraints)), simplify(Or(unbounded_constraints))


# Case: For each action a → l(s,a) == {} AND l(t, a) == {}
# Result: A[s][t] = 0
def postfixpoint_2(spa: DeterministicSPA, d: Array, s: str, t: str):
    return d[spa.states.index(s)][spa.states.index(t)] == 0


# Case: If there's an action a: l(s,a) == {} AND l(t,a) != {} or vice versa
# Result: A[s][t] = 1
def postfixpoint_3(spa: DeterministicSPA, d: Array, s: str, t: str):
    return d[spa.states.index(s)][spa.states.index(t)] == 1
