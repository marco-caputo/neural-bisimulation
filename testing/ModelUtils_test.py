import unittest
import torch
import tensorflow as tf
from NNToGraph import *

TORCH_MODEL_1 = TorchFFNN([2, 1])
TORCH_MODEL_2 = TorchFFNN([10, 12, 14, 7])
TF_MODEL_1 = TensorFlowFFNN([3, 2])
TF_MODEL_2 = TensorFlowFFNN([5, 13, 16, 8, 9])

class ModelUtilsTest(unittest.TestCase):
    def test_layers_torch_1(self):
        ls = list(layers(TORCH_MODEL_1))
        self.assertEqual(len(ls), 1)
        self.assertEqual(type(ls[0]), torch.nn.Linear)

        ls = list(layers(TORCH_MODEL_1, layer_type=torch.nn.Linear))
        self.assertEqual(type(ls[0]), torch.nn.Linear)

    def test_layers_torch_2(self):
        ls = list(layers(TORCH_MODEL_2))
        self.assertEqual(len(ls), 5)
        self.assertTrue(type(ls[2]) == torch.nn.Linear)
        self.assertTrue(type(ls[3]) == torch.nn.ReLU)

        ls = list(layers(TORCH_MODEL_2, layer_type=torch.nn.Linear))
        self.assertEqual(len(ls), 3)
        self.assertTrue(all([type(l) == torch.nn.Linear for l in ls]))

        ls = list(layers(TORCH_MODEL_2, layer_type=torch.nn.ReLU))
        self.assertEqual(len(ls), 2)
        self.assertTrue(all([type(l) == torch.nn.ReLU for l in ls]))

    def test_layers_tf_1(self):
        ls = list(layers(TF_MODEL_1))
        self.assertEqual(len(ls), 1)
        self.assertEqual(type(ls[0]), tf.keras.layers.Dense)

        ls = list(layers(TF_MODEL_1, layer_type=tf.keras.layers.Dense))
        self.assertEqual(type(ls[0]), tf.keras.layers.Dense)

    def test_layers_tf_2(self):
        ls = list(layers(TF_MODEL_2))
        self.assertEqual(len(ls), 7)
        self.assertTrue(type(ls[2]) == tf.keras.layers.Dense)
        self.assertTrue(type(ls[3]) == tf.keras.layers.ReLU)

        ls = list(layers(TF_MODEL_2, layer_type=tf.keras.layers.Dense))
        self.assertEqual(len(ls), 4)
        self.assertTrue(all([type(l) == tf.keras.layers.Dense for l in ls]))

        ls = list(layers(TF_MODEL_2, layer_type=tf.keras.layers.ReLU))
        self.assertEqual(len(ls), 3)
        self.assertTrue(all([type(l) == tf.keras.layers.ReLU for l in ls]))

    def test_input_dim_torch_1(self):
        self.assertEqual(input_dim(TORCH_MODEL_1), 2)

    def test_input_dim_torch_2(self):
        self.assertEqual(input_dim(TORCH_MODEL_2), 10)

    def test_output_dim_torch_1(self):
        self.assertEqual(output_dim(TORCH_MODEL_1), 1)

    def test_output_dim_torch_2(self):
        self.assertEqual(output_dim(TORCH_MODEL_2), 7)

    def test_input_dim_tf_1(self):
        model = TensorFlowFFNN([3, 2])
        self.assertEqual(input_dim(TF_MODEL_1), 3)

    def test_input_dim_tf_2(self):
        self.assertEqual(input_dim(TF_MODEL_2), 5)

    def test_output_dim_tf_1(self):
        model = TensorFlowFFNN([3, 2])
        self.assertEqual(output_dim(TF_MODEL_1), 2)

    def test_output_dim_tf_2(self):
        self.assertEqual(output_dim(TF_MODEL_2), 9)


if __name__ == '__main__':
    unittest.main()
