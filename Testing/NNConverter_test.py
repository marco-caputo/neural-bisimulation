import unittest

from NeuralNetworks import *
from ApproxBisimulation import FiniteStateProcess

NN_MODEL_1 = NeuralNetwork([[[-1, 1], [0.5, 0.5], [0.5, 0.5]]], [[0,0]], [ReLU()])

class NeuralNetworkTest(unittest.TestCase):

    def test_to_fsp(self):
        fsp = to_fsp(NN_MODEL_1)
        self.assertEqual(6, len(fsp.states))
        self.assertEqual({ACTION, FiniteStateProcess.TAU}, fsp.all_actions())
        self.assertEqual({"x1", "x2", "x3"}, fsp.target_states("s", ACTION))
        self.assertEqual({"h1_1", "h1_2"}, fsp.target_states("x2", ACTION))
        self.assertEqual({"h1_2"}, fsp.target_states("x1", ACTION))