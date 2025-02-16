import bispy as bp
import networkx as nx
from z3 import *

from Testing import *
from NeuralNetworks.Graphs import *
from SMTEquivalence import *


def bisimulation_test():
    dgraph = nx.DiGraph()
    dgraph.add_edge(1, 2)
    dgraph.add_edge(2, 1)
    dgraph.add_edge(2, 3)
    dgraph.add_edge(1, 4)

    print(bp.compute_maximum_bisimulation(dgraph, [(1, 2), (3, 4)]))

def model_visualization_test():
    layers_dim = [2, 3, 4, 1]  # Input: 2 neurons, Hidden1: 3 neurons, Hidden2: 4 neurons, Output: 2 neurons
    model = TorchFFNN(layers_dim)  # TensorFlowFFNN(layers_dim)
    graph = create_graph(model, add_biases=True)
    visualize_model_graph(graph, inter_layer_distance=2.0, intra_layer_distance=0.5, round_digits=4)

def z3_test():
    """
    # Declare variables
    x = Real('x')
    y = Real('y')

    # Define the function to be optimized
    f = 3 * x - 2 * y

    feasible, opt_sol = get_optimal_solution(f, [x >= 0, y >= 0, x <= 1, y <= 1], maximize=True)
    feasible, opt_sol = get_optimal_solution(f, [x >= 0, y >= 0, x <= 1, y <= 1], maximize=False)

    worst_case = Real('worst_case')  # Proxy variable to store the worst-case value

    # Create the Z3 optimization solver
    opt = Optimize()

    # Define the function to be optimized
    f = 3 * x - 2 * y

    # Inner maximization: maximize f(x, y) and store result in 'worst_case'
    opt.add(worst_case == f)  # worst_case captures the worst-case outcome
    opt.maximize(worst_case)  # Maximize with respect to y

    # Outer minimization: minimize the worst-case outcome with respect to x
    #opt.minimize(worst_case)

    # Add constraints
    opt.add(x >= 0, y >= 0)

    # Solve the nested optimization problem
    if opt.check() == sat:
        model = opt.model()
        print(f"Optimal x: {model[x]}")
        print(f"Optimal y: {model[y]}")
        print(f"Optimal f: {model.eval(f)}")
        print(f"Optimal function value: {model[worst_case]}")
    else:
        print("No solution found.")
    """

    """
    # Declare variables
    x = Real('x')
    y = Real('y')

    # Define the function to be optimized
    f = 3 * x - 2 * y
    constraints = [x >= 0, y >= 0, x <= 3, y >= x, y <= 5]
    _, opt_x_val = get_optimal_assignment(f, constraints, x, maximize=True)
    print(opt_x_val)
    constraints.append(x == opt_x_val)
    print(get_optimal_solution(f, constraints, maximize=False))

    
    f = 3 * x - 2 * y
    constraints = [x >= 0, y >= 0, y >= x]
    _, opt_x_val = get_optimal_assignment(f, constraints, x, maximize=True)
    print(opt_x_val)
    constraints.append(x == opt_x_val)
    print(get_optimal_solution(f, constraints, maximize=False))

    # Declare variables
    x, y, d = Reals('x y d')

    # Define the function f(x, y, d)
    f = 3 * x - 2 * y + d

    # Create an optimization solver
    opt = Optimize()

    # Constraints
    opt.add(x >= 0, x <= 3)
    opt.add(y >= 0, y <= 5)
    opt.add(d >= -10, d <= 10)  # d is existentially quantified

    # Minimize over x, Maximize over y
    inner_max = opt.maximize(f)  # max_y f(x, y, d)
    outer_min = opt.minimize(inner_max)  # min_x max_y f(x, y, d)

    # Solve for d such that the min-max problem is feasible
    exists_d = Exists([d], opt.check() == sat)

    if exists_d:
        model = opt.model()
        print(f"Optimal d: {model[d]}")
        print(f"Optimal function value: {model.evaluate(f)}")
    else:
        print("No solution exists for d.")
    """
    d = Real('d')

    x = Real('x')
    y = Real('y')
    f = 3 * x - 2 * y
    constraints = [x >= 0, y >= 0, x <= y, x + y <= d]

    x2 = Real('x2')
    y2 = Real('y2')
    f2 = 3 * x2 - 2 * y2
    constraints2 = [x2 >= 0, y2 >= 0, x2 <= y2, x2 + y2 <= d]

    formula1, opt_sol1 = get_optimization_problem_formula(f, constraints, [x, y], maximize=True, suffix="f1")
    formula2, opt_sol2 = get_optimization_problem_formula(f2, constraints2, [x2, y2], maximize=True, suffix="f2")

    formula3, opt_sol3 = get_optimization_problem_formula(opt_sol1,
                                                [opt_sol1 == opt_sol2, formula1, formula2, d>=0],
                                                [d], maximize=True, suffix="f3")

    """
    a = Real('a')
    b = Real('b')
    formula_test = Exists([a], ForAll([b], a >= b))

    print(formula_test)

    s = Solver()
    s.add(formula_test)
    print(s.check())
    model = s.model()  # model is not available (empty)
    print(float(model[a].as_decimal(10)))

    """


    print(formula3)

    s = Solver()
    s.add(formula3)
    print(s.check())
    model = s.model() # model is not available (empty)
    print(float(model[opt_sol3].as_decimal(10)))

    s.add(opt_sol3 > 1e16)
    print(s.check())
    model = s.model()  # model is not available (empty)
    print(float(model[opt_sol3].as_decimal(10)))



def main():
    # bisimulation_test()
    # model_visualization_test()
    z3_test()


if __name__ == "__main__":
    main()
