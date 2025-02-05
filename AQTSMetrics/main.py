from AQTSMetrics.DistanceAlgorithm import distance_approx
from AQTSMetrics.SpaModel import SPA


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

spa = SPA(spa1)
print (distance_approx(spa, 0.1, 's1', 't2'))




################# To print the satisfiable d_matrix #################
#result = solver.check()
#print(result)
#model = solver.model()
#ctr = [[model.eval(Xarr[i][j]) for j in range(7)] for i in range(7)]
#print(ctr)