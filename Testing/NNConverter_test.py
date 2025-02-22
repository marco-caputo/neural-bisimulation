import unittest

from NeuralNetworks import *
from ApproxBisimulation import ProbabilisticFiniteStateProcess

NN_MODEL_1 = NeuralNetwork([[[-1, 1], [0.5, 0.5], [0.5, 0.5]]], [[0,0]], [ReLU()])
NN_MODEL_2 = NeuralNetwork([[[-1, 1], [0.5, 0.5], [0.6, 0.4]]], [[0,0]], [ReLU()])
NN_MODEL_3 = NeuralNetwork([[[-1, 1], [0.5, 0.5], [0.6, 0.4]]], [[1,0]], [ReLU()])

NN_MODEL_4 = NeuralNetwork([[[1, 1]]*3, [[-1, 2], [1, -2]]], [[0, 0], [0, 0]], [ReLU(), ReLU()])
NN_MODEL_5 = NeuralNetwork([[[0, 0], [0, 0], [0, 1]], [[0, 0], [1, 0]]], [[0, 0], [0, 0]], [Identity(), Identity()])
NN_MODEL_6 = NeuralNetwork([[[1, 0], [0, 0], [0, 1]], [[0, 1], [1, -1]]], [[0, 0], [0, 0]], [ReLU(), ReLU()])

SEED = 1234

class NNConverterTest(unittest.TestCase):

    def test_to_fsp_1(self):
        fsp = to_fsp(NN_MODEL_1, [(0, float("inf"))]*3)
        self.assertEqual(6, len(fsp.states))
        self.assertEqual({ACTION, ProbabilisticFiniteStateProcess.TAU}, fsp.all_actions())
        self.assertEqual({"x0": 1/3, "x1": 1/3, "x2": 1/3}, fsp.target_states(START_STATE, ACTION)[0])

        for x in ["x0", "x1", "x2"]:
            self.assertEqual({"h1_1": 1.0}, fsp.target_states(x, ACTION)[0])

    def test_to_fsp_2(self):
        fsp = to_fsp(NN_MODEL_2, [(0, float("inf"))]*3)
        self.assertEqual({"x0": 1/3, "x1": 1/3, "x2": 1/3}, fsp.target_states("s", ACTION)[0])
        self.assertEqual({"h1_1": 1.0}, fsp.target_states("x0", ACTION)[0])
        self.assertEqual({"h0_1": 0.5, "h1_1": 0.5}, fsp.target_states("x1", ACTION)[0])
        self.assertEqual({"h0_1": 0.5, "h1_1": 0.5}, fsp.target_states("x2", ACTION)[0])

    def test_to_fsp_3(self):
        fsp = to_fsp(NN_MODEL_3, [(0, 0.5)]*3)
        self.assertEqual({"x0": 1/3, "x1": 1/3, "x2": 1/3}, fsp.target_states("s", ACTION)[0])

        for x in ["x0", "x1", "x2"]:
            self.assertEqual({"h0_1": 1.0}, fsp.target_states(x, ACTION)[0])

    def test_to_fsp_4_shallow_bounds(self):
        fsp = to_fsp(NN_MODEL_4, [(0, 1)]*3, shallow_bounds=True)
        self.assertEqual({"x0": 1/3, "x1": 1/3, "x2": 1/3}, fsp.target_states("s", ACTION)[0])

        for s in ["x0", "x1", "x2"]:
            self.assertEqual(list(), fsp.target_states(s, ACTION))
        self.assertEqual({"h1_2": 1.0}, fsp.target_states("h0_1", ACTION)[0])
        self.assertEqual({"h0_2": 1.0}, fsp.target_states("h1_1", ACTION)[0])

    def test_to_fsp_4_not_shallow_bounds(self):
        fsp = to_fsp(NN_MODEL_4, [(0, 1)]*3, shallow_bounds=False)
        self.assertEqual({"x0": 1/3, "x1": 1/3, "x2": 1/3}, fsp.target_states("s", ACTION)[0])

        for s in ["x0", "x1", "x2", "h0_1", "h1_1"]:
            self.assertEqual(list(), fsp.target_states(s, ACTION))

    def test_to_spa_probabilistic_1(self):
        spa = to_spa_probabilistic(NN_MODEL_1, lower=0, seed=SEED)
        self.assertEqual(6, len(spa.states))
        self.assertEqual([ACTION], spa.labels)

        for x in ["x0", "x1", "x2"]:
            self.assertNotIn("h0_1", spa.distribution(x, ACTION))
            self.assertEqual(1.0, spa.get_probability(x, ACTION, "h1_1"))

    def test_to_spa_probabilistic_2(self):
        spa = to_spa_probabilistic(NN_MODEL_1, seed=SEED)
        self.assertTrue(spa.get_probability("x0", ACTION, "h1_1") > spa.get_probability("x0", ACTION, "h0_1"))
        self.assertTrue(spa.get_probability("x1", ACTION, "h1_1") < spa.get_probability("x1", ACTION, "h0_1"))
        self.assertTrue(spa.get_probability("x2", ACTION, "h1_1") < spa.get_probability("x2", ACTION, "h0_1"))

    def test_to_spa_probabilistic_3(self):
        spa = to_spa_probabilistic(NN_MODEL_3, lower=0, upper=0.5, seed=SEED)
        for x in ["x0", "x1", "x2"]:
            self.assertNotIn("h1_1", spa.distribution(x, ACTION))
            self.assertEqual(1.0, spa.get_probability(x, ACTION, "h0_1"))

    def test_to_spa_probabilistic_4(self):
        spa = to_spa_probabilistic(NN_MODEL_4, lower=0, upper=1, seed=SEED)
        self.assertEqual(8, len(spa.states))
        self.assertEqual([ACTION], spa.labels)

        for x in ["x0", "x1", "x2"]:
            self.assertNotIn("h1_1", spa.distribution(x, ACTION))
            self.assertEqual(1.0, spa.get_probability(x, ACTION, "h0_1"))

        self.assertNotIn("h1_2", spa.distribution("h0_1", ACTION))
        self.assertEqual(1.0, spa.get_probability("h0_1", ACTION, "h0_2"))
        self.assertEqual({}, spa.distribution("h1_1", ACTION))

    def test_to_spa_probabilistic_5(self):
        spa = to_spa_probabilistic(NN_MODEL_5, seed=SEED)

        self.assertTrue(spa.get_probability("x0", ACTION, "h0_1") > spa.get_probability("x0", ACTION, "h1_1"))
        self.assertTrue(spa.get_probability("x1", ACTION, "h0_1") > spa.get_probability("x1", ACTION, "h1_1"))
        self.assertTrue(spa.get_probability("x2", ACTION, "h1_1") > spa.get_probability("x2", ACTION, "h0_1"))

        self.assertEqual(1.0, spa.get_probability("h0_1", ACTION, "h1_2"))
        self.assertEqual(1.0, spa.get_probability("h1_1", ACTION, "h0_2"))

    def test_to_spa_probabilistic_6(self):
        spa = to_spa_probabilistic(NN_MODEL_6, lower=0, seed=SEED)

        self.assertEqual(0.0, spa.get_probability("x0", ACTION, "h1_1"))
        self.assertEqual(0.0, spa.get_probability("x2", ACTION, "h0_1"))
        for trans in [("x0", "h0_1"), ("x1", "h0_1"), ("x1", "h1_1"), ("x2", "h1_1")]:
            self.assertTrue(spa.get_probability(trans[0], ACTION, trans[1]) > 0)

        self.assertEqual(0.0, spa.get_probability("h1_1", ACTION, "h1_2"))
        for trans in [("h0_1", "h0_2"), ("h0_1", "h1_2"), ("h1_1", "h0_2")]:
            self.assertTrue(spa.get_probability(trans[0], ACTION, trans[1]) > 0)






