import unittest
from typing import List, Tuple

from NeuralNetworks import *

TORCH_MODEL_1 = TorchFFNN([2, 1])
TORCH_MODEL_2 = TorchFFNN([10, 12, 14, 7])
TF_MODEL_1 = TensorFlowFFNN([3, 2])
TF_MODEL_2 = TensorFlowFFNN([5, 13, 16, 8, 9])


class ModelUtilsTest(unittest.TestCase):

    def assertFloatsEqual(self, a: float, b: float):
        self.assertTrue(abs(a - b) < 1e-6)

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
        print(ls)
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

    def test_get_layer_tensor_shape_torch_1(self):
        ls = list(layers(TORCH_MODEL_1, layer_type=torch.nn.Linear))
        self.assertEqual(len(get_layer_tensor(ls[0])), 2)
        self.assertEqual(len(get_layer_tensor(ls[0])[0]), 1)

    def test_get_layer_tensor_shape_torch_2(self):
        ls = list(layers(TORCH_MODEL_2, layer_type=torch.nn.Linear))
        self.assertEqual(len(get_layer_tensor(ls[1])), 12)
        self.assertEqual(len(get_layer_tensor(ls[1])[0]), 14)

    def test_get_layer_tensor_shape_tf_1(self):
        ls = list(layers(TF_MODEL_1, layer_type=tf.keras.layers.Dense))
        self.assertEqual(len(get_layer_tensor(ls[0])), 3)
        self.assertEqual(len(get_layer_tensor(ls[0])[0]), 2)

    def test_get_layer_tensor_shape_tf_2(self):
        ls = list(layers(TF_MODEL_2, layer_type=tf.keras.layers.Dense))
        self.assertEqual(len(get_layer_tensor(ls[3])), 8)
        self.assertEqual(len(get_layer_tensor(ls[3])[0]), 9)

    def test_get_layer_tensor_values_torch_1(self):
        model = clone_model(TORCH_MODEL_1)
        set_weights_on_layer(model.layers[0], [[0.1], [0.2]], [0.3])
        ls = list(layers(model, layer_type=torch.nn.Linear))
        self.assertFloatsEqual(get_layer_tensor(ls[0])[0][0], 0.1)
        self.assertFloatsEqual(get_layer_tensor(ls[0])[1][0], 0.2)

    def test_get_layer_tensor_values_torch_2(self):
        model = clone_model(TORCH_MODEL_2)
        set_weights_on_layer(model.layers[0],
                             [[0.1] * 12, [0.2] * 12, [0.3] * 12, [0.4] * 12, [0.5] * 12, [0.6] * 12, [0.7] * 12,
                              [0.8] * 12, [0.9] * 12, [1.0] * 12],
                             [0.1] * 12)
        ls = list(layers(model, layer_type=torch.nn.Linear))
        self.assertFloatsEqual(get_layer_tensor(ls[0])[1][1], 0.2)
        self.assertFloatsEqual(get_layer_tensor(ls[0])[7][11], 0.8)

    def test_get_layer_tensor_values_tf_1(self):
        model = clone_model(TF_MODEL_1)
        set_weights_on_layer(model.layers[0], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [0.7, 0.8])
        ls = list(layers(model, layer_type=tf.keras.layers.Dense))
        self.assertFloatsEqual(get_layer_tensor(ls[0])[0][1], 0.2)
        self.assertFloatsEqual(get_layer_tensor(ls[0])[2][1], 0.6)

    def test_get_layer_tensor_values_tf_2(self):
        model = clone_model(TF_MODEL_2)
        set_weights_on_layer(list(layers(model, layer_type=tf.keras.layers.Dense))[3],
                             [[0.1] * 9, [0.2] * 9, [0.3] * 9, [0.4] * 9, [0.5] * 9, [0.6] * 9, [0.7] * 9, [0.8] * 9],
                             [0.1] * 9)
        ls = list(layers(model, layer_type=tf.keras.layers.Dense))
        self.assertFloatsEqual(get_layer_tensor(ls[3])[1][1], 0.2)
        self.assertFloatsEqual(get_layer_tensor(ls[3])[6][8], 0.7)

    def test_get_layer_biases_torch_1(self):
        model = clone_model(TORCH_MODEL_1)
        set_weights_on_layer(model.layers[0], [[0.1], [0.1]], [0.3])
        ls = list(layers(model, layer_type=torch.nn.Linear))
        self.assertFloatsEqual(get_layer_biases(ls[0])[0], 0.3)

    def test_get_layer_biases_tf_1(self):
        model = clone_model(TF_MODEL_1)
        set_weights_on_layer(model.layers[0], [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]], [0.7, 0.7])
        ls = list(layers(model, layer_type=tf.keras.layers.Dense))
        self.assertFloatsEqual(get_layer_biases(ls[0])[1], 0.7)

    def test_apply_to_tensors_torch_1(self):
        model = clone_model(TORCH_MODEL_1)
        set_weights_on_layer(model.layers[0], [[0.1], [0.1]], [0.3])
        apply_to_tensors(model, weight_proc=lambda w, i, j, k: self.assertFloatsEqual(w, 0.1),
                         bias_proc=lambda b, i, j: self.assertFloatsEqual(b, 0.3))

    def test_apply_to_tensors_torch_2(self):
        model = clone_model(TORCH_MODEL_2)
        set_weights_on_layer(model.layers[0],
                             [[0.1] * 12, [0.2] * 12, [0.3] * 12, [0.4] * 12, [0.5] * 12, [0.6] * 12, [0.7] * 12,
                              [0.8] * 12, [0.9] * 12, [1.0] * 12],
                             [0.1] * 12)
        weights_calls: List[Tuple[float, int, int, int]] = []
        biases_calls: List[Tuple[float, int, int]] = []
        apply_to_tensors(model, weight_proc=lambda w, i, j, k: weights_calls.append((w, i, j, k)),
                         bias_proc=lambda b, i, j: biases_calls.append((b, i, j)))

        self.assertEqual(len(weights_calls), 10 * 12 + 12 * 14 + 14 * 7)
        self.assertEqual(len(biases_calls), sum([12, 14, 7]))

        self.assertFloatsEqual(weights_calls[0][0], 0.1)
        self.assertEqual(weights_calls[0], (weights_calls[0][0], 1, 0, 0))

        self.assertFloatsEqual(weights_calls[3 * 12 + 1][0], 0.4)
        self.assertEqual(weights_calls[3 * 12 + 1], (weights_calls[3 * 12 + 1][0], 1, 3, 1))

        self.assertFloatsEqual(biases_calls[3][0], 0.1)
        self.assertEqual(biases_calls[3], (biases_calls[3][0], 1, 3))

    def test_apply_to_tensors_tf_1(self):
        model = clone_model(TF_MODEL_1)
        set_weights_on_layer(model.layers[0], [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]], [0.7, 0.7])
        apply_to_tensors(model, weight_proc=lambda w, i, j, k: self.assertFloatsEqual(w, 0.1),
                         bias_proc=lambda b, i, j: self.assertFloatsEqual(b, 0.7))

    def test_apply_to_tensors_tf_2(self):
        model = clone_model(TF_MODEL_2)
        set_weights_on_layer(list(layers(model, layer_type=tf.keras.layers.Dense))[3],
                             [[0.1] * 9, [0.2] * 9, [0.3] * 9, [0.4] * 9, [0.5] * 9, [0.6] * 9, [0.7] * 9, [0.8] * 9],
                             [0.1] * 9)
        weights_calls: List[Tuple[float, int, int, int]] = []
        biases_calls: List[Tuple[float, int, int]] = []
        apply_to_tensors(model, weight_proc=lambda w, i, j, k: weights_calls.append((w, i, j, k)),
                         bias_proc=lambda b, i, j: biases_calls.append((b, i, j)))

        self.assertEqual(len(weights_calls), 5 * 13 + 13 * 16 + 16 * 8 + 8 * 9)
        self.assertEqual(len(biases_calls), sum([13, 16, 8, 9]))

        index = 5 * 13 + 13 * 16 + 16 * 8
        self.assertFloatsEqual(weights_calls[index][0], 0.1)
        self.assertEqual(weights_calls[index], (weights_calls[index][0], 4, 0, 0))

        index = 5 * 13 + 13 * 16 + 16 * 8 + 3 * 9 + 1
        self.assertFloatsEqual(weights_calls[index][0], 0.4)
        self.assertEqual(weights_calls[index], (weights_calls[index][0], 4, 3, 1))

        self.assertFloatsEqual(biases_calls[sum([13, 16, 8])][0], 0.1)
        self.assertEqual(biases_calls[sum([13, 16, 8]) + 3], (biases_calls[sum([13, 16, 8]) + 3][0], 4, 3))


if __name__ == '__main__':
    unittest.main()
