import unittest

from NeuralNetworks import *
from ApproxBisimulation import FiniteStateProcess

NN_MODEL_1 = NeuralNetwork([[[-1, 1], [0.5, 0.5], [0.5, 0.5]]], [[0,0]], [ReLU()])
NN_MODEL_2 = NeuralNetwork([[[-1, 1], [0.5, 0.5], [0.6, 0.4]]], [[0,0]], [ReLU()])
NN_MODEL_3 = NeuralNetwork([[[-1, 1], [0.5, 0.5], [0.6, 0.4]]], [[1,0]], [ReLU()])

NN_MODEL_4 = NeuralNetwork([[[1, 1]]*3, [[-1, 2], [1, -2]]], [[0, 0], [0, 0]], [ReLU(), ReLU()])


class NNConverterTest(unittest.TestCase):

    def test_to_fsp_1(self):
        fsp = to_fsp(NN_MODEL_1, [(0, float("inf"))]*3)
        self.assertEqual(6, len(fsp.states))
        self.assertEqual({ACTION, FiniteStateProcess.TAU}, fsp.all_actions())
        self.assertEqual({"x0", "x1", "x2"}, fsp.target_states("s", ACTION))
        self.assertEqual({"h1_1"}, fsp.target_states("x0", ACTION))
        self.assertEqual({"h1_1"}, fsp.target_states("x1", ACTION))
        self.assertEqual({"h1_1"}, fsp.target_states("x2", ACTION))

    def test_to_fsp_2(self):
        fsp = to_fsp(NN_MODEL_2, [(0, float("inf"))]*3)
        self.assertEqual({"x0", "x1", "x2"}, fsp.target_states("s", ACTION))
        self.assertEqual({"h1_1"}, fsp.target_states("x0", ACTION))
        self.assertEqual({"h0_1", "h1_1"}, fsp.target_states("x1", ACTION))
        self.assertEqual({"h0_1", "h1_1"}, fsp.target_states("x2", ACTION))

    def test_to_fsp_3(self):
        fsp = to_fsp(NN_MODEL_3, [(0, 0.5)]*3)
        self.assertEqual({"x0", "x1", "x2"}, fsp.target_states("s", ACTION))
        self.assertEqual({"h0_1"}, fsp.target_states("x0", ACTION))
        self.assertEqual({"h0_1"}, fsp.target_states("x1", ACTION))
        self.assertEqual({"h0_1"}, fsp.target_states("x2", ACTION))

    def test_tp_fsp_4_shallow_bounds(self):
        fsp = to_fsp(NN_MODEL_4, [(0, 1)]*3, shallow_bounds=True)
        self.assertEqual({"x0", "x1", "x2"}, fsp.target_states("s", ACTION))
        self.assertEqual(set(), fsp.target_states("x0", ACTION))
        self.assertEqual(set(), fsp.target_states("x1", ACTION))
        self.assertEqual(set(), fsp.target_states("x2", ACTION))
        self.assertEqual({"h1_2"}, fsp.target_states("h0_1", ACTION))
        self.assertEqual({"h0_2"}, fsp.target_states("h1_1", ACTION))

    def test_tp_fsp_4_not_shallow_bounds(self):
        fsp = to_fsp(NN_MODEL_4, [(0, 1)]*3, shallow_bounds=False)
        self.assertEqual({"x0", "x1", "x2"}, fsp.target_states("s", ACTION))
        self.assertEqual(set(), fsp.target_states("x0", ACTION))
        self.assertEqual(set(), fsp.target_states("x1", ACTION))
        self.assertEqual(set(), fsp.target_states("x2", ACTION))
        self.assertEqual(set(), fsp.target_states("h0_1", ACTION))
        self.assertEqual(set(), fsp.target_states("h1_1", ACTION))


