from PostFixPoint import *


def distance_approx(spa:SpaModel, epsilon:Real, s:String, t:String):
    lower_bound = 0
    upper_bound = 1
    m = 0.5
    d_matrix = Array('DistanceMatrix', IntSort(), ArraySort(IntSort(), RealSort()))
    solver = Solver()
    for i in range(math.ceil(math.log(1 / epsilon, 2))):
        solver.reset()
        print(f'ITERATION {i}')
        solver.add(d_matrix[spa.states.index(s)][spa.states.index(t)] <= m)
        solver.add(pseudo(d_matrix, len(spa.states)))
        for state_1 in spa.states:
            for state_2 in spa.states:
                if any((spa.squiggly_l(s, a) ^ spa.squiggly_l(t, a)) for a in spa.labels):
                    solver.add(postfixpoint_3(d_matrix, state_1, state_2, spa))
                if all(spa.squiggly_l(s, a) or spa.squiggly_l(t, a) for a in spa.labels):
                    solver.add(postfixpoint_2(d_matrix, state_1, state_2, spa))
                if all(not (spa.squiggly_l(s, a) ^ spa.squiggly_l(t, a)) for a in spa.labels):
                    solver.add(postfixpoint_1(spa, d_matrix, state_1, state_2, f'{state_1}_{state_2}_{i}'))
        result = solver.check()
        if result != z3.unsat and result != z3.unknown:
            upper_bound = m
        else:
            lower_bound = m
        m = (upper_bound + lower_bound) / 2
    return [lower_bound, upper_bound]


def pseudo(d_matrix:Array[IntSort(), ArraySort(IntSort(), RealSort())], n_states:Int):
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
    return constraints