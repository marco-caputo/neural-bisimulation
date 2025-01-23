import unittest

from z3 import *
from NNToGraph import *
from SMTEquivalence import *

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

    def test_encode_into_SMT_formula_torch_1(self):
        smt, inp, outp = encode_into_SMT_formula(TORCH_MODEL_1)
        weights = next(tensors(TORCH_MODEL_1))
        bias = next(biases(TORCH_MODEL_1))
        expected_formula = weights[0][0] * inp[0] + weights[1][0] * inp[1] + bias[0] == outp[0]
        s = Solver()
        s.add(smt != expected_formula)
        self.assertEqual(s.check(), unsat)

    def test_encode_torch_2(self):
        smt, inp, outp = encode_into_SMT_formula(TORCH_MODEL_2)
        self.assertEqual(len(inp), 3)
        self.assertEqual(len(outp), 2)
        self.assertTrue(get_float_formula_satisfiability(smt, inp)[0])

    def test_encode_tf_1(self):
        smt, inp, outp = encode_into_SMT_formula(TF_MODEL_1)
        self.assertEqual(len(inp), 3)
        self.assertEqual(len(outp), 2)
        self.assertTrue(get_float_formula_satisfiability(smt, inp)[0])


    def test_encode_into_SMT_formula_tf_1(self):
        smt, inp, outp = encode_into_SMT_formula(TF_MODEL_1)
        weights = next(tensors(TF_MODEL_1))
        bias = next(biases(TF_MODEL_1))
        expected_formula = And(weights[0][0] * inp[0] + weights[1][0] * inp[1] + weights[2][0] * inp[2] + bias[0] == outp[0],
                               weights[0][1] * inp[0] + weights[1][1] * inp[1] + weights[2][1] * inp[2] + bias[1] == outp[1])
        s = Solver()
        s.add(smt != expected_formula)
        self.assertEqual(s.check(), unsat)

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