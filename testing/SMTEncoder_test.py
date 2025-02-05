import unittest

from z3 import *
from NNToGraph import *
from SMTEquivalence import *
import torch.nn as nn

import os

TORCH_MODEL_1 = TorchFFNN([2, 1])
TORCH_MODEL_2 = TorchFFNN([3, 4, 5, 2])
TF_MODEL_1 = TensorFlowFFNN([3, 2])
TF_MODEL_2 = TensorFlowFFNN([2, 4, 5, 5, 3])

class ModelUtilsTest(unittest.TestCase):

    def assertFloatsEqual(self, a: float, b: float):
        self.assertTrue(abs(a - b) < 1e-6)

    def test_get_formula_sat(self):
        inputs = [Real('x1'), Real('x2'), Real('x3')]
        formula = inputs[0] + inputs[1] + inputs[2] > 0

        res = get_float_formula_satisfiability(formula, inputs)
        self.assertTrue(res[0])
        self.assertTrue(sum(res[1]) > 0)

    def test_encode_torch_1(self):
        smt, inp, outp = encode_into_SMT_formula(TORCH_MODEL_1)
        self.assertEqual(len(inp), 2)
        self.assertEqual(len(outp), 1)
        self.assertTrue(get_float_formula_satisfiability(smt, inp)[0])

    def test_encode_tf_1(self):
        smt, inp, outp = encode_into_SMT_formula(TF_MODEL_1)
        self.assertEqual(len(inp), 3)
        self.assertEqual(len(outp), 2)
        self.assertTrue(get_float_formula_satisfiability(smt, inp)[0])

    def test_encode_into_SMT_formula_torch(self):
        smt, inp, outp = encode_into_SMT_formula(TORCH_MODEL_1)
        weights = next(tensors(TORCH_MODEL_1))
        bias = next(biases(TORCH_MODEL_1))
        expected_formula = weights[0][0] * inp[0] + weights[1][0] * inp[1] + bias[0] == outp[0]
        s = Solver()
        s.add(smt != expected_formula)
        self.assertEqual(s.check(), unsat)

    def test_encode_into_SMT_formula_tf(self):
        smt, inp, outp = encode_into_SMT_formula(TF_MODEL_1)
        weights = next(tensors(TF_MODEL_1))
        bias = next(biases(TF_MODEL_1))
        expected_formula = And(weights[0][0] * inp[0] + weights[1][0] * inp[1] + weights[2][0] * inp[2] + bias[0] == outp[0],
                               weights[0][1] * inp[0] + weights[1][1] * inp[1] + weights[2][1] * inp[2] + bias[1] == outp[1])
        s = Solver()
        s.add(smt != expected_formula)
        self.assertEqual(s.check(), unsat)

    def test_encode_torch_2(self):
        smt, inp, outp = encode_into_SMT_formula(TORCH_MODEL_2)
        self.assertEqual(len(inp), 3)
        self.assertEqual(len(outp), 2)
        self.assertTrue(get_float_formula_satisfiability(smt, inp)[0])

    def test_encode_tf_2(self):
        smt, inp, outp = encode_into_SMT_formula(TF_MODEL_2)
        self.assertEqual(len(inp), 2)
        self.assertEqual(len(outp), 3)
        self.assertTrue(get_float_formula_satisfiability(smt, inp)[0])

    def test_equivalent_models_torch_1(self):
        are_equivalent, counterexample = are_strict_equivalent(
            TORCH_MODEL_1,
            clone_model(TORCH_MODEL_1)
        )
        self.assertTrue(are_equivalent)
        self.assertIsNone(counterexample)

    def test_equivalent_models_tf_1(self):
        are_equivalent, counterexample = are_strict_equivalent(
            TF_MODEL_1,
            clone_model(TF_MODEL_1)
        )
        self.assertTrue(are_equivalent)
        self.assertIsNone(counterexample)

    def test_equivalent_models_torch_2(self):
        model1 = clone_model(TORCH_MODEL_1)
        weights = [[-2.45], [0.64]]
        bias = [1.2]
        set_weights_on_layer(model1.layers_list[0], weights, bias)

        model2 = TorchFFNN([2, 2, 1])
        model2.layers_list[1] = nn.Identity()
        set_weights_on_layer(model2.layers_list[0], [[2,1.5], [-1,-1.2]], [1, 2])
        set_weights_on_layer(model2.layers_list[2], [[-2.2], [1.3]], [0.8])

        are_equivalent, counterexample = are_strict_equivalent(model1, model2, [(-1, 1)]*2)
        self.assertTrue(are_equivalent)
        self.assertIsNone(counterexample)

    def test_equivalent_models_tf_2(self):
        model1 = clone_model(TF_MODEL_1)
        weights = [[0.05, 0.1], [0.15, 0.1], [-0.02, 0.14]]
        bias = [0.23, 0.14]
        set_weights_on_layer(model1.layers_list[0], weights, bias)

        model2 = TensorFlowFFNN([3, 2, 2])
        model2.layers_list[1] = nn.Identity()
        set_weights_on_layer(model2.layers_list[0], [[0.1, 0.2], [-0.3, 0.4], [0.5, 0.1]], [-0.5, 0.6])
        set_weights_on_layer(model2.layers_list[2], [[-0.1,  0.2], [0.3, 0.4]], [0, 0])

        are_equivalent, counterexample = are_strict_equivalent(model1, model2, [(-1, 1)]*3)
        self.assertTrue(are_equivalent)
        self.assertIsNone(counterexample)

    def test_non_equivalent_models_torch_1(self):
        unequal_model = clone_model(TORCH_MODEL_1)
        weights = next(tensors(TORCH_MODEL_1))
        weights[1][0] += 10.0
        set_weights_on_layer(unequal_model.layers_list[0], weights, next(biases(TORCH_MODEL_1)))
        are_equivalent, counterexample = are_strict_equivalent(
            TORCH_MODEL_1,
            unequal_model
        )
        self.assertFalse(are_equivalent)
        self.assertIsNotNone(counterexample)
        self.assertEqual(len(counterexample), 2)

    def test_non_equivalent_models_tf_1(self):
        unequal_model = clone_model(TF_MODEL_1)
        weights = next(tensors(TF_MODEL_1))
        weights[1][0] += 1.0
        set_weights_on_layer(unequal_model.layers_list[0], weights, next(biases(TF_MODEL_1)))
        are_equivalent, counterexample = are_strict_equivalent(
            TF_MODEL_1,
            unequal_model
        )
        self.assertFalse(are_equivalent)
        self.assertIsNotNone(counterexample)
        self.assertEqual(len(counterexample), 3)

    def test_non_equivalent_models_torch_2(self):
        model1 = clone_model(TORCH_MODEL_1)
        weights = [[-2.46], [0.64]]  # -2.46 instead of -2.45 for the first weight
        bias = [1.2]
        set_weights_on_layer(model1.layers_list[0], weights, bias)

        model2 = TorchFFNN([2, 2, 1])
        model2.layers_list[1] = nn.Identity()
        set_weights_on_layer(model2.layers_list[0], [[2,1.5], [-1,-1.2]], [1, 2])
        set_weights_on_layer(model2.layers_list[2], [[-2.2], [1.3]], [0.8])

        are_equivalent, counterexample = are_strict_equivalent(model1, model2, [(-1, 1)]*2)
        self.assertFalse(are_equivalent)
        self.assertIsNotNone(counterexample)
        self.assertEqual(len(counterexample), 2)

    def test_non_equivalent_models_tf_2(self):
        model1 = clone_model(TF_MODEL_1)
        weights = [[0.05, 0.1], [0.16, 0.1], [-0.02, 0.14]] # 0.16 instead of 0.15 for the first weight
        bias = [0.23, 0.14]
        set_weights_on_layer(model1.layers_list[0], weights, bias)

        model2 = TensorFlowFFNN([3, 2, 2])
        model2.layers_list[1] = nn.Identity()
        set_weights_on_layer(model2.layers_list[0], [[0.1, 0.2], [-0.3, 0.4], [0.5, 0.1]], [-0.5, 0.6])
        set_weights_on_layer(model2.layers_list[2], [[-0.1, 0.2], [0.3, 0.4]], [0, 0])

        are_equivalent, counterexample = are_strict_equivalent(model1, model2, [(-1, 1)] * 3)
        self.assertFalse(are_equivalent)
        self.assertIsNotNone(counterexample)
        self.assertEqual(len(counterexample), 3)

    # Tests weather two models from different frameworks are considered equivalent whenever they have the same
    # architecture and weights
    def test_equivalent_models_torch_and_tensorflow(self):
        tf_model = clone_model(TF_MODEL_1)
        weights = next(tensors(TF_MODEL_1))
        bias = next(biases(TF_MODEL_1))

        torch_model = TorchFFNN([3, 2])
        set_weights_on_layer(torch_model.layers_list[0], weights, bias)
        are_equivalent, counterexample = are_strict_equivalent(
            torch_model,
            tf_model
        )
        self.assertTrue(are_equivalent)
        self.assertIsNone(counterexample)
