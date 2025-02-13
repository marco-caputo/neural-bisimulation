from z3 import *
from AQTSMetrics import SpaModel


# Base Case: For each action a → l(s,a) != {} ↔ l(t,a) != {}
# Result:
def postfixpoint_1(spa:SpaModel, d_matrix:Array, s:String, t:String, variable_name:String):
    y = Int(f'{variable_name}_p1y')
    z = Int(f'{variable_name}_p1z')
    print(f'State {s}, State {t}')
    labels = spa.data[s].keys()
    return Exists(y, And(Or([haus(y, spa, d_matrix, s, t, a, f'{y}_{a}') for a in labels]),
               ForAll(z, Implies(Or([haus(z, spa, d_matrix, s, t, a, f'{z}_{a}') for a in labels]), y >= z))))


def haus(y:Int, spa:SpaModel, d_matrix:Array, s:String, t:String, a:String, variable_name:String):
    z = Int(f'{variable_name}_hausz')
    return And(sup_inf(y, spa, d_matrix, s, t, a, f'{y}_s_t'),
               sup_inf(y, spa, d_matrix, t, s, a, f'{y}_t_s'),
               ForAll(z, Implies(And(sup_inf(z, spa, d_matrix, s, t, a, f'{z}_s_t'),
                                     sup_inf(y, spa, d_matrix, t, s, a, f'{z}_t_s')), y >= z)))


def sup_inf(y:Int, spa:SpaModel, d_matrix:Array, s:String, t:String, a:String, variable_name:String):
    z = Int(f'{variable_name}_sup_infz')
    return And(Or([inf(y, spa, d_matrix, t, a, tau, f'{y}_{tau}') for tau in spa.squiggly_l(s, a)]),
               ForAll(z, Or([Implies(inf(z, spa, d_matrix, t, a, tau, f'{z}_{tau}'), y>=z) for tau in spa.squiggly_l(s, a)])))


def inf(y:Int, spa:SpaModel, d_matrix:Array, t:String, a:String, tau:dict[String, Real], variable_name:String):
    z = Int(f'{variable_name}_infz')
    return And(Or([hd(y, spa, d_matrix, tau, tau1, f'{y}_{tau1}') for tau1 in spa.squiggly_l(t, a)]),
               ForAll(z, Or([Implies(hd(z, spa, d_matrix, tau, tau1, f'{z}_{tau1}'), y <= z) for tau1 in spa.squiggly_l(t, a)])))


def hd(y:Int, spa:SpaModel, d_matrix:Array, tau:dict[String, Real], tau1:dict[String, Real], variable_name:String):
    z = Int(f'{variable_name}_hdz')
    return And(lp_squiggly(y, spa, d_matrix, tau, tau1, f'{y}_{tau}_{tau1}'),
               ForAll(z, Implies(lp_squiggly(z, spa, d_matrix, tau, tau1, f'{z}_{tau}_{tau1}'), y >= z)))


def lp_squiggly(y:Int, spa:SpaModel, d_matrix:Array, tau:dict[String, Real], tau1:dict[String, Real], variable_name:String):
    x_array = Array(f'{variable_name}_array', IntSort(), RealSort())
    constraints = []
    for s in spa.states:
        constraints.append(0 <= x_array[spa.states.index(s)])
        constraints.append(x_array[spa.states.index(s)] <= 1)
        for t in spa.states:
            constraints.append(
                x_array[spa.states.index(s)] - x_array[spa.states.index(t)] <= d_matrix[spa.states.index(s)][spa.states.index(t)])
    constraints.append(y == Sum([(tau.get(s, 0.) - tau1.get(s, 0.)) * x_array[spa.states.index(s)] for s in spa.states]))
    return Exists(x_array, And(constraints))


# Case: For each action a → l(s,a) == {} AND l(t, a) == {}
# Result: A[s][t] = 0
def postfixpoint_2(spa:SpaModel, d_matrix:Array, s:String, t:String):
    return d_matrix[spa.states.index(s)][spa.states.index(t)] == 0


# Case: If there's an action a: l(s,a) == {} AND l(t,a) != {} or vice versa
# Result: A[s][t] = 1
def postfixpoint_3(spa:SpaModel, d_matrix:Array, s:String, t:String):
    return d_matrix[spa.states.index(s)][spa.states.index(t)] == 1
