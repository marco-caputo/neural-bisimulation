from AQTSMetrics import DeterministicSPA
from PostFixPointDet import *


def distance_approx_det(spa: DeterministicSPA, epsilon: float, s: str, t: str):
    lower_bound = 0
    upper_bound = 1
    m = 0.5
    d = Array('DistanceMatrix', IntSort(), ArraySort(IntSort(), RealSort()))
    solver = Solver()
    for i in range(math.ceil(math.log(1 / epsilon, 2))):
        solver.reset()
        print(f'ITERATION {i}')
        solver.add(d[spa.index_of(s)][spa.index_of(t)] <= m)
        solver.add(pseudo(d, len(spa.states)))
        for i, s1 in enumerate(spa.states):
            for j, s2 in enumerate(spa.states, start=i):
                if any((bool(spa.distribution(s1, a)) ^ bool(spa.distribution(s2, a))) for a in spa.labels):
                    solver.add(postfixpoint_3(spa, d, s1, s2))
                elif all(not (bool(spa.distribution(s1, a)) or bool(spa.distribution(s2, a))) for a in spa.labels):
                    solver.add(postfixpoint_2(spa, d, s1, s2))
                elif all(not (bool(spa.distribution(s1, a)) ^ bool(spa.distribution(s2, a))) for a in spa.labels):
                    solver.add(postfixpoint_1(spa, d, s1, s2))

        result = solver.check()
        if result != z3.sat:
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