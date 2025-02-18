from AQTSMetrics.DistanceAlgorithm import distance_approx
from AQTSMetrics.DistanceAlgorithmDet import distance_approx_det
from AQTSMetrics.SpaModel import SPA, DeterministicSPA

spa1 = {
    's1': {'c': [ {'t1': 0.5,
                   't2': 0.5} ]},
    's2': {'c': [ {'t1': 0.2, 't2': 0.8},
                    {'t3': 0.4, 't4': 0.6} ]},
    't1': {'a': [ {'z0': 1.} ]},
    't2': {'b': [ {'z0': 1.} ]},
    't3': {'a': [ {'z0': 1.} ]},
    't4': {'b': [ {'z0': 1.} ]},
    'z0': {}
}

spa2 = {
    's1': {'c': [ {'t1': 0.5, 't2': 0.5}, {'t3': 0.4, 't4': 0.6} ]},
    's2': {'c': [ {'t1': 0.5, 't2': 0.5},
                    {'t3': 0.4, 't4': 0.6} ]},
    't1': {},
    't2': {},
    't3': {},
    't4': {'b': [ {'z0': 1.} ]},
    'z0': {}
}

spa3 = {
    's1': {'c': {'t1': 0.5, 't2': 0.5}},
    's2': {'c': {'t1': 0.2, 't2': 0.3, 't3': 0.2, 't4': 0.3}},
    't1': {'a': {'z0': 1.}},
    't2': {'b': {'z0': 1.}},
    't3': {'a': {'z0': 1.}},
    't4': {'b': {'z0': 1.}},
    'z0': {}
}

spa4 = {
    's': {'b': {'t1': 0.25, 't2': 0.25, 't3': 0.25, 't4': 0.25}},
    't1': {'a': {'h1': 0.23, 'h2': 0.37, 'h3': 0.4}},
    't2': {'a': {'h1': 0.15, 'h2': 0.35, 'h3': 0.5}},
    't3': {'a': {'h1': 0.5, 'h2': 0.3, 'h3': 0.2}},
    't4': {'a': {'h1': 0.1, 'h2': 0.2, 'h3': 0.7}},
    'h1': {'a': {'y1': 0.4, 'y2': 0.6}},
    'h2': {'a': {'y1': 0.33, 'y2': 0.67}},
    'h3': {'a': {'y1': 0.8, 'y2': 0.2}},
    'y1': {},
    'y2': {},

    'sc': {'b': {'t1c': 0.25, 't2c': 0.25, 't3c': 0.25, 't4c': 0.25}},
    't1c': {'a': {'h1c': 0.239, 'h2c': 0.36, 'h3c': 0.401}},
    't2c': {'a': {'h1c': 0.25, 'h2c': 0.25, 'h3c': 0.5}},
    't3c': {'a': {'h1c': 0.51, 'h2c': 0.3, 'h3c': 0.19}},
    't4c': {'a': {'h1c': 0.1, 'h2c': 0.2, 'h3c': 0.7}},
    'h1c': {'a': {'y1c': 0.4, 'y2c': 0.6}},
    'h2c': {'a': {'y1c': 0.53, 'y2c': 0.47}},
    'h3c': {'a': {'y1c': 0.8, 'y2c': 0.2}},
    'y1c': {},
    'y2c': {}
}

det_spa = DeterministicSPA(spa3)
print(distance_approx_det(det_spa, 0.01, 's1', 's2'))

#spa = SPA(spa1)
#print(distance_approx(spa, 0.01, 's1', 's2'))




################# To print the satisfiable d_matrix #################
#result = solver.check()
#print(result)
#model = solver.model()
#ctr = [[model.eval(Xarr[i][j]) for j in range(7)] for i in range(7)]
#print(ctr)