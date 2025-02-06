import unittest

from z3 import *
from NNToGraph import *
from SMTEquivalence import *
import torch.nn as nn

import os

TORCH_MODEL_1 = TorchFFNN([2, 1])
TORCH_MODEL_2 = TorchFFNN([3, 4, 5, 2])
TF_MODEL_1 = TensorFlowFFNN([3, 2])
TF_MODEL_2 = TensorFlowFFNN([2, 5, 5, 3])

class ModelUtilsTest(unittest.TestCase):

    def get_equivalent_torch_models(self):
        model1 = clone_model(TORCH_MODEL_1)
        weights = [[-2.45], [0.64]]
        bias = [1.2]
        set_weights_on_layer(model1.layers[0], weights, bias)

        model2 = TorchFFNN([2, 2, 1], activation_layer=nn.Identity())
        set_weights_on_layer(model2.layers[0], [[2, 1.5], [-1, -1.2]], [1, 2])
        set_weights_on_layer(model2.layers[2], [[-2.2], [1.3]], [0.8])

        return model1, model2

    def get_equivalent_tf_models(self):
        model1 = clone_model(TF_MODEL_1)
        weights = [[0.05, 0.1], [0.15, 0.1], [-0.02, 0.14]]
        bias = [0.23, 0.14]
        set_weights_on_layer(model1.layers[0], weights, bias)

        model2 = TensorFlowFFNN([3, 2, 2], activation_layer=tf.keras.layers.Identity())
        set_weights_on_layer(model2.layers[0], [[0.1, 0.2], [-0.3, 0.4], [0.5, 0.1]], [-0.5, 0.6])
        set_weights_on_layer(model2.layers[2], [[-0.1, 0.2], [0.3, 0.4]], [0, 0])

        return model1, model2

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

    # Tests the encoding of a model with different activation functions
    def test_encode_torch_3(self):
        model = clone_model(TORCH_MODEL_2)
        model.layers[1] = nn.Hardsigmoid()
        model.layers[3] = nn.Hardtanh()

        smt, inp, outp = encode_into_SMT_formula(model)
        self.assertEqual(len(inp), 3)
        self.assertEqual(len(outp), 2)
        self.assertTrue(get_float_formula_satisfiability(smt, inp)[0])

    def test_encode_tf_3(self):
        model = clone_model(TF_MODEL_2)
        model.layers[1] = tf.keras.layers.Activation(activation=tf.keras.activations.hard_sigmoid)
        model.layers[3] = tf.keras.layers.LeakyReLU(negative_slope=0.5)

        smt, inp, outp = encode_into_SMT_formula(model)
        self.assertEqual(len(inp), 2)
        self.assertEqual(len(outp), 3)
        self.assertTrue(get_float_formula_satisfiability(smt, inp)[0])

    def test_different_input_dims(self):
        model1 = TorchFFNN([2, 1])
        model2 = TorchFFNN([3, 1])
        self.assertRaises(ValueError, are_strict_equivalent, model1, model2)

    def test_different_output_dims(self):
        model1 = TensorFlowFFNN([2, 1])
        model2 = TensorFlowFFNN([2, 2])
        self.assertRaises(ValueError, are_strict_equivalent, model1, model2)

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
        model1, model2 = self.get_equivalent_torch_models()

        are_equivalent, counterexample = are_strict_equivalent(model1, model2, [(-1, 1)]*2)
        self.assertTrue(are_equivalent)
        self.assertIsNone(counterexample)

    def test_equivalent_models_tf_2(self):
        model1, model2 = self.get_equivalent_tf_models()

        are_equivalent, counterexample = are_strict_equivalent(model1, model2, [(-1, 1)]*3)
        self.assertTrue(are_equivalent)
        self.assertIsNone(counterexample)

    def test_non_equivalent_models_torch_1(self):
        unequal_model = clone_model(TORCH_MODEL_1)
        weights = next(tensors(TORCH_MODEL_1))
        weights[1][0] += 10.0
        set_weights_on_layer(unequal_model.layers[0], weights, next(biases(TORCH_MODEL_1)))
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
        set_weights_on_layer(unequal_model.layers[0], weights, next(biases(TF_MODEL_1)))
        are_equivalent, counterexample = are_strict_equivalent(
            TF_MODEL_1,
            unequal_model
        )
        self.assertFalse(are_equivalent)
        self.assertIsNotNone(counterexample)
        self.assertEqual(len(counterexample), 3)

    # Tests weather two models from pytorch are not considered strictly equivalent whenever they have a
    # difference in the weights of the first layer compared with the original equivalent model
    def test_non_equivalent_models_torch_2(self):
        model1, model2 = self.get_equivalent_torch_models()
        weights = next(tensors(model1))
        weights[0][0] -= 0.1 # -2.46 instead of -2.45 for the first weight
        set_weights_on_layer(model1.layers[0], weights, next(biases(model1)))

        are_equivalent, counterexample = are_strict_equivalent(model1, model2, [(-1, 1)]*2)
        self.assertFalse(are_equivalent)
        self.assertIsNotNone(counterexample)
        self.assertEqual(len(counterexample), 2)

    # Tests weather two models from tensor flow are not considered strictly equivalent whenever they have a
    # difference in the weights of the first layer compared with the original equivalent model
    def test_non_equivalent_models_tf_2(self):
        model1, model2 = self.get_equivalent_tf_models()
        weights = next(tensors(model1))
        weights[1][0] += 0.01 # 0.16 instead of 0.15 for the first weight
        set_weights_on_layer(model1.layers[0], weights, next(biases(model1)))

        are_equivalent, counterexample = are_strict_equivalent(model1, model2, [(-1, 1)] * 3)
        self.assertFalse(are_equivalent)
        self.assertIsNotNone(counterexample)
        self.assertEqual(len(counterexample), 3)

    # Tests weather two models from different frameworks are considered equivalent whenever they have the same
    # architecture and weights
    def test_equivalent_models_torch_and_tf(self):
        tf_model = clone_model(TF_MODEL_1)
        weights = next(tensors(TF_MODEL_1))
        bias = next(biases(TF_MODEL_1))

        torch_model = TorchFFNN([3, 2])
        set_weights_on_layer(torch_model.layers[0], weights, bias)
        are_equivalent, counterexample = are_strict_equivalent(
            torch_model,
            tf_model
        )
        self.assertTrue(are_equivalent)

    def test_approximate_equivalence_torch_1(self):
        are_equivalent, counterexample = are_approximate_equivalent(
            TORCH_MODEL_1,
            clone_model(TORCH_MODEL_1),
            [(-1, 1)]*2
        )
        self.assertTrue(are_equivalent)
        self.assertIsNone(counterexample)

    def test_approximate_equivalence_tf_1(self):
        are_equivalent, counterexample = are_approximate_equivalent(
            TF_MODEL_1,
            clone_model(TF_MODEL_1),
            [(-1, 1)]*3,
            p=2,
            epsilon=1e-9
        )
        self.assertTrue(are_equivalent)

    def test_approximate_equivalence_torch_2(self):
        model1, model2 = self.get_equivalent_torch_models()

        are_equivalent, counterexample = are_approximate_equivalent(
            model1,
            model2,
            [(-1, 1)]*2,
            p=2
        )
        self.assertTrue(are_equivalent)

    def test_approximate_equivalence_tf_2(self):
        model1, model2 = self.get_equivalent_tf_models()

        are_equivalent, counterexample = are_approximate_equivalent(
            model1,
            model2,
            [(-1, 1)]*3
        )
        self.assertTrue(are_equivalent)

    # Tests weather two models from pytorch are considered approximately equivalent whenever they have
    # weights that lead to a contained difference in the output of the model
    def test_approximate_equivalence_torch_3(self):
        weights_1, weights_2 = [[1], [1]], [[1.1], [1.1]]
        bias_1, bias_2 = [1], [1]
        model1, model2 = clone_model(TORCH_MODEL_1), clone_model(TORCH_MODEL_1)
        set_weights_on_layer(model1.layers[0], weights_1, bias_1)
        set_weights_on_layer(model2.layers[0], weights_2, bias_2)

        are_equivalent, counterexample = are_approximate_equivalent(model1, model2, [(-1, 1)]*2, p=1, epsilon=0.21)
        self.assertTrue(are_equivalent)

        are_equivalent, counterexample = are_approximate_equivalent(model1, model2, [(-1, 1)]*2, p=1, epsilon=0.19)
        self.assertFalse(are_equivalent)

    def test_approximate_equivalence_tf_3(self):
        weights_1, weights_2 = [[1, 1], [1, 1], [1, 1]], [[1.1, 1.1], [1.1, 1.1], [1.1, 1.1]]
        bias_1, bias_2 = [0, 0], [0, 0]
        model1, model2 = clone_model(TF_MODEL_1), clone_model(TF_MODEL_1)
        set_weights_on_layer(model1.layers[0], weights_1, bias_1)
        set_weights_on_layer(model2.layers[0], weights_2, bias_2)

        are_equivalent, counterexample = are_approximate_equivalent(model1, model2, [(0, 1)]*3, p=2, epsilon=0.5)
        self.assertTrue(are_equivalent)

        are_equivalent, counterexample = are_approximate_equivalent(model1, model2, [(0, 1)]*3, p=2, epsilon=0.4)
        self.assertFalse(are_equivalent)


    #Tests weather models from different frameworks are considered approximately equivalent whenever they have the same
    #architecture and activation function different from ReLu.
    def test_approximate_equivalence_torch_and_tf(self):
        weights = [[[0.5, 0.1], [-0.15, 0.2], [-0.02, 1.00], [0.1, 0.3]], [[-0.5, 0.2], [0.35, -0.1]]]
        bias = [[0.4, -0.3], [0.2, 0.1]]
        layers_dim = [4, 2, 2]
        model1 = TorchFFNN(layers_dim, activation_layer=nn.Hardsigmoid())
        model2 = TensorFlowFFNN(layers_dim, activation_layer=tf.keras.layers.Activation(activation=tf.keras.activations.hard_sigmoid))
        set_weights_on_layer(model1.layers[0], weights[0], bias[0])
        set_weights_on_layer(model2.layers[0], weights[0], bias[0])
        set_weights_on_layer(model1.layers[2], weights[1], bias[1])
        set_weights_on_layer(model2.layers[2], weights[1], bias[1])

        are_equivalent, counterexample = are_approximate_equivalent(
            model1,
            model2,
            [(-1, 1)]*4,
            p=float('inf')
        )
        self.assertTrue(are_equivalent)

    def test_argmax_equivalence_torch_1(self):
        are_equivalent, counterexample = are_argmax_equivalent(
            TORCH_MODEL_1,
            clone_model(TORCH_MODEL_1)
        )
        self.assertTrue(are_equivalent)

    def test_argmax_equivalence_tf_1(self):
        are_equivalent, counterexample = are_argmax_equivalent(
            TF_MODEL_1,
            clone_model(TF_MODEL_1)
        )
        self.assertTrue(are_equivalent)


    # Tests weather two models from pytorch are considered equivalent whenever one of the two outputs is
    # always greater than the other
    def test_argmax_equivalence_tf_2(self):
        weights_1, weights_2 = [[2, 1], [2, 1], [2, 1]], [[10, 1.1], [10, 1.1], [10, 1.1]]
        bias_1, bias_2 = [1, 1], [1, 1]
        model1, model2 = clone_model(TF_MODEL_1), clone_model(TF_MODEL_1)
        set_weights_on_layer(model1.layers[0], weights_1, bias_1)
        set_weights_on_layer(model2.layers[0], weights_2, bias_2)

        are_equivalent, counterexample = are_argmax_equivalent(model1, model2)
        self.assertTrue(are_equivalent)

        changed_weights = [[1, 1.1], [1, 1.1], [1, 1.1]]
        set_weights_on_layer(model2.layers[0], changed_weights, bias_2)

        are_equivalent, counterexample = are_argmax_equivalent(model1, model2)
        self.assertFalse(are_equivalent)

    def test_argmax_equivalence_tf_3(self):
        model1 = TorchFFNN([3, 2], activation_layer=nn.Hardsigmoid())
        model2 = TorchFFNN([3, 2, 2], activation_layer=nn.Hardsigmoid())
        weights1, weights21, weights22 = [[2, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]], [[2, 1], [1, 1]]
        bias = [0, 0]
        set_weights_on_layer(model1.layers[0], weights1, bias)
        set_weights_on_layer(model2.layers[0], weights21, bias)
        set_weights_on_layer(model2.layers[2], weights22, bias)

        are_equivalent, counterexample = are_argmax_equivalent(model1, model2, [(0, 1)]*3)
        print(counterexample)
        self.assertTrue(are_equivalent)

        weights = next(tensors(model1))
        weights[0][0] -= 2
        set_weights_on_layer(model1.layers[0], weights, next(biases(model1)))
        are_equivalent, counterexample = are_argmax_equivalent(model1, model2, [(0, 1)]*3)
        self.assertFalse(are_equivalent)

