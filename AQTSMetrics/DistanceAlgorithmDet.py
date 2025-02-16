import z3

from AQTSMetrics import DeterministicSPA
from PostFixPointDet import *


def distance_approx_det(spa: DeterministicSPA, epsilon: float, s: str, t: str):
    lower_bound = 0
    upper_bound = 1
    m = 0.5
    solver = Solver()
    solver.set("timeout", 15000)
    for i in range(math.ceil(math.log(1 / epsilon, 2))):
        sat_constraints = []
        unb_constraints = []
        print(f'ITERATION {i}')
        d = Array(f'DistanceMatrix_{i}', IntSort(), ArraySort(IntSort(), RealSort()))
        for j, s1 in enumerate(spa.states):
            for s2 in spa.states[j:]:
                if any((bool(spa.distribution(s1, a)) ^ bool(spa.distribution(s2, a))) for a in spa.labels):
                    sat_constraints.append(postfixpoint_3(spa, d, s1, s2))
                elif all(not (bool(spa.distribution(s1, a)) or bool(spa.distribution(s2, a))) for a in spa.labels):
                    sat_constraints.append(postfixpoint_2(spa, d, s1, s2))
                elif all(not (bool(spa.distribution(s1, a)) ^ bool(spa.distribution(s2, a))) for a in spa.labels):
                    satisfiable, unbounded = postfixpoint_1(spa, d, s1, s2)
                    sat_constraints.append(satisfiable)
                    unb_constraints.append(unbounded)

        solver.reset()
        solver.add(pseudo(d, len(spa.states)))
        solver.add(d[spa.index_of(s)][spa.index_of(t)] <= m)
        solver.add(simplify(And(sat_constraints)))
        result_sat = solver.check()
        if result_sat == z3.unknown:
            print(solver.reason_unknown())
        solver.reset()
        solver.add(pseudo(d, len(spa.states)))
        solver.add(d[spa.index_of(s)][spa.index_of(t)] <= m)
        solver.add(simplify(Or(unb_constraints)))
        result_unb = solver.check()
        print(result_sat, result_unb)

        if result_unb == z3.sat or result_sat != z3.sat:
            lower_bound = m
        else:
            upper_bound = m
        m = (upper_bound + lower_bound) / 2

    return [lower_bound, upper_bound]


def pseudo(d_matrix: Array, n_states: int) -> BoolRef:
    """
    Returns the Z3 constraint expressing the pseudo-metric property of the given distance matrix variable.
    """
    constraints = []
    for s in range(n_states):
        constraints.append(d_matrix[s][s] == 0)
        for t in range(n_states):
            if s != t:
                constraints.append(d_matrix[s][t] == d_matrix[t][s])
                constraints.append(d_matrix[s][t] <= 1)
                constraints.append(d_matrix[s][t] >= 0)
                for w in range(n_states):
                    constraints.append(d_matrix[s][t] + d_matrix[t][w] >= d_matrix[s][w])
    return And(constraints)