from typing import List, Tuple, Optional
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
        return True, [float(model.evaluate(x).as_decimal(10).rstrip('?')) for x in
                      inputs]  # Forse bastava fare model[x].as_long()
    elif result == unsat:
        return False, None
    else:
        raise RuntimeError("Solver returned 'unknown'. The equivalence might be too complex to decide.")


def is_satisfiable(formula: BoolRef) -> bool:
    """
    Checks the satisfiability of a Z3 formula and returns a boolean indicating if the formula is satisfiable.

    :param formula: A Z3 formula
    :return: A boolean indicating if the formula is satisfiable
    """
    s = Solver()
    s.add(formula)
    if s.check() == unknown:
        raise RuntimeError("Solver returned 'unknown'. The equivalence might be too complex to decide.")
    return s.check() == sat


def get_optimal_solution(objective: ArithRef, constraints: list[BoolRef], maximize: bool = True) -> Tuple[
    bool, float | ArithRef | None]:
    """
    Finds the optimal value of the given float objective function under the given constraints.

    The output of the function is a tuple containing a boolean indicating if the optimization problem is feasible and
    the optimal value. In particular:
    - If the optimization problem has an empty feasible region, the function returns False and None;
    - If the optimization problem is unbounded, the function returns True and None;
    - If the optimization problem is feasible and bounded, the function returns True and the optimal float value.

    :param objective: The objective function to optimize
    :param constraints: A list of constraints
    :param maximize: A boolean indicating whether to maximize or minimize the objective
    :return: A tuple containing a boolean indicating if the optimization problem is feasible and the optimal value if any
    """
    opt = Optimize()
    opt.add(And(constraints))

    obj = opt.maximize(objective) if maximize else opt.minimize(objective)
    if opt.check() == sat:
        b = opt.upper(obj) if maximize else opt.lower(obj)
        if str(b) in {"oo", "-1*oo"}:
            return True, None
        else:
            return True, b.as_long() if b.is_int() else float(b.as_decimal(10).rstrip('?'))

    return False, None


def get_optimal_assignment(objective: ArithRef, constraints: list[BoolRef], variable: ArithRef,
                           maximize: bool = True) -> \
        Tuple[bool, Optional[float]]:
    """
    Finds the optimal value of the given objective function under the given constraints
    and retrieves the assigned value of a specified variable in an optimal solution.

    The output of the function is a tuple containing:
    1. A boolean indicating if the optimization problem is feasible.
    2. The optimal value of the objective function (or None if unbounded/infeasible).
    3. The optimal value of the specified variable (or None if unbounded/infeasible).

    :param objective: The objective function to optimize.
    :param constraints: A list of constraints.
    :param variable: The variable whose value we want in an optimal solution.
    :param maximize: A boolean indicating whether to maximize or minimize the objective.
    :return: A tuple (is_feasible, optimal_value, optimal_variable_value).
    """
    opt = Optimize()
    opt.add(And(constraints))

    obj = opt.maximize(objective) if maximize else opt.minimize(objective)

    if opt.check() == sat:
        model = opt.model()
        optimal_value = opt.upper(obj) if maximize else opt.lower(obj)

        if str(optimal_value) in {"oo", "-oo"}:
            return True, model[variable]  # Unbounded solution

        # Retrieve the assigned value of the given variable in an optimal solution
        optimal_var_value = model[variable]
        optimal_var_value = optimal_var_value.as_long() if optimal_var_value.is_int() else float(
            optimal_var_value.as_decimal(
                10).rstrip('?'))

        return True, optimal_var_value

    return False, None  # Infeasible solution


def get_optimization_problem_formula(objective: ArithRef,
                                     constraints: list[BoolRef],
                                     free_variables: list[Real] | list[Array],
                                     maximize: bool = True,
                                     suffix: str = "") -> tuple[BoolRef, Real]:
    """
    Converts an optimization problem into a SMT formula without using optimize, so the formula can be used in
    more complex SMT problems.

    The output of the function is a tuple containing:
    - The SMT formula representing the optimization problem, evaluated as true if an optimal solution exists.
    - A Z3 variable representing the optimal value of the objective function.

    :param objective: The objective function to optimize
    :param constraints: A list of constraints
    :param free_variables: A list of free variables in the optimization problem
    :param maximize: A boolean indicating whether to maximize or minimize the objective
    :param suffix: A suffix to add to the names of the variables in the produced SMT formula
    :return: A Z3 formula representing the optimization problem
    """
    opt_sol = Real(f'opt{suffix}')
    """
    sol = Real(f'sol{suffix}')
    return And(
        Exists(free_variables, And(opt_sol == objective, And(constraints))),
        ForAll([sol], Implies(Exists(free_variables, And(sol == objective, And(constraints))),
                              (sol <= opt_sol if maximize else sol >= opt_sol)))
    ), opt_sol
    """
    """
    return And(opt_sol == objective,
            And(constraints),
            ForAll(free_variables, Implies(
                And(constraints),
                (objective <= opt_sol if maximize else objective >= opt_sol)))), opt_sol
    """
    if isinstance(free_variables[0], ArrayRef):
        free_variables_copy = [_copy_array(x) for x in free_variables]
    else:
        free_variables_copy = [Real(f'{x}_copy') for x in free_variables]

    substitutions = [(x, free_variables_copy[i]) for i, x in enumerate(free_variables)]
    constraints_copy = substitute(And(constraints), substitutions)
    objective_copy = substitute(objective, substitutions)

    return And(opt_sol == objective,
               And(constraints),
               Not(Exists(free_variables_copy, And(
                   constraints_copy,
                   objective_copy > opt_sol if maximize else objective_copy < opt_sol)))), opt_sol


def _copy_array(arr: ArrayRef):
    domain_type = arr.sort().domain()
    range_type = arr.sort().range()

    # Handle matrices (arrays of arrays)
    if isinstance(range_type, ArraySortRef):
        range_domain_type = range_type.domain()  # Column index type
        range_range_type = range_type.range()  # Element type (e.g., Real, Int, etc.)

        # Recreate the matrix (nested arrays)
        return Array(f'{arr}_copy', domain_type, ArraySort(range_domain_type, range_range_type))
    else:
        # Single array
        return Array(f'{arr}_copy', domain_type, range_type)
