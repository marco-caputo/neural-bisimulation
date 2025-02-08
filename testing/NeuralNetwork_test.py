import unittest
from NeuralNetworks import *

TORCH_MODEL_1 = TorchFFNN([2, 1])
TORCH_MODEL_2 = TorchFFNN([10, 12, 14, 2],
                          activation_layer=torch.nn.Hardshrink(),
                          output_activation=torch.nn.Hardsigmoid())
TF_MODEL_1 = TensorFlowFFNN([3, 2])
TF_MODEL_2 = TensorFlowFFNN([5, 6, 7, 4, 3],
                            activation_func=tf.keras.layers.ReLU())
TF_MODEL_3 = TensorFlowFFNN([5, 6, 7, 3],
                            activations_as_layers=False)
TF_MODEL_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(6, activation=tf.keras.activations.hard_sigmoid),
    tf.keras.layers.Dense(3, activation=None)
])
TF_MODEL_5 = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(6),
    tf.keras.layers.Activation(tf.keras.activations.hard_sigmoid),
    tf.keras.layers.Dense(3)
])

class NeuralNetworkTest(unittest.TestCase):
    def test_from_torch_model_1(self):
        nn = NeuralNetwork.from_model(TORCH_MODEL_1)
        self.assertEqual(nn.size(), 1)
        self.assertEqual(nn.layers[0].input_size(), 2)
        self.assertEqual(nn.layers[0].output_size(), 1)
        self.assertTrue(isinstance(nn.layers[0].activation_functions[0], Identity))

    def test_forward_pass_torch_model_1(self):
        nn = NeuralNetwork.from_model(TORCH_MODEL_1)
        nn.layers[0].weights = [[2], [1.5]]
        nn.layers[0].biases = [3]
        input = [1.0, 2.0]
        self.assertEqual(nn(input), [8])

    def test_from_tf_model_1(self):
        nn = NeuralNetwork.from_model(TF_MODEL_1)
        self.assertEqual(nn.size(), 1)
        self.assertEqual(nn.layers[0].input_size(), 3)
        self.assertEqual(nn.layers[0].output_size(), 2)
        self.assertTrue(isinstance(nn.layers[0].activation_functions[0], Identity))

    def test_forward_pass_tf_model_1(self):
        nn = NeuralNetwork.from_model(TF_MODEL_1)
        nn.layers[0].weights = [[1, 2], [3, 4], [-5, -6]]
        nn.layers[0].biases = [10, 5]
        nn.layers[0].activation_functions = [ReLU(), ReLU()]
        input = [1, 2, 3]
        self.assertEqual(nn(input), [2, 0])

    def test_from_torch_model_2(self):
        nn2 = NeuralNetwork.from_model(TORCH_MODEL_2)
        self.assertEqual(nn2.size(), 3)
        self.assertEqual(nn2.layers[0].input_size(), 10)
        self.assertEqual(nn2.layers[0].output_size(), 12)
        self.assertEqual(nn2.layers[1].input_size(), 12)
        self.assertEqual(nn2.layers[1].output_size(), 14)
        self.assertEqual(nn2.layers[2].input_size(), 14)
        self.assertEqual(nn2.layers[2].output_size(), 2)

        self.assertTrue(isinstance(nn2.layers[0].activation_functions[0], HardShrink))
        self.assertTrue(isinstance(nn2.layers[1].activation_functions[5], HardShrink))
        self.assertTrue(isinstance(nn2.layers[-1].activation_functions[1], HardSigmoid))

    def test_from_tf_model_2(self):
        nn2 = NeuralNetwork.from_model(TF_MODEL_2)
        self.assertEqual(nn2.size(), 4)
        self.assertEqual(nn2.layers[0].input_size(), 5)
        self.assertEqual(nn2.layers[0].output_size(), 6)
        self.assertEqual(nn2.layers[1].input_size(), 6)
        self.assertEqual(nn2.layers[1].output_size(), 7)
        self.assertEqual(nn2.layers[2].input_size(), 7)
        self.assertEqual(nn2.layers[2].output_size(), 4)
        self.assertEqual(nn2.layers[3].input_size(), 4)
        self.assertEqual(nn2.layers[3].output_size(), 3)

        self.assertTrue(isinstance(nn2.layers[0].activation_functions[0], ReLU))
        self.assertTrue(isinstance(nn2.layers[2].activation_functions[3], ReLU))
        self.assertTrue(isinstance(nn2.layers[-1].activation_functions[1], Identity))

    def test_from_tf_model_3(self):
        nn3 = NeuralNetwork.from_model(TF_MODEL_3)
        self.assertEqual(nn3.size(), 3)
        self.assertEqual(nn3.layers[0].input_size(), 5)
        self.assertEqual(nn3.layers[0].output_size(), 6)
        self.assertEqual(nn3.layers[1].input_size(), 6)
        self.assertEqual(nn3.layers[1].output_size(), 7)
        self.assertEqual(nn3.layers[2].input_size(), 7)
        self.assertEqual(nn3.layers[2].output_size(), 3)

        self.assertTrue(isinstance(nn3.layers[0].activation_functions[0], ReLU))
        self.assertTrue(isinstance(nn3.layers[2].activation_functions[1], Identity))

    def test_from_tf_model_4(self):
        dummy_input = tf.random.uniform((1, 3))
        TF_MODEL_4(dummy_input) # Build the model

        nn4 = NeuralNetwork.from_model(TF_MODEL_4)
        self.assertEqual(nn4.size(), 3)
        self.assertEqual(nn4.layers[0].input_size(), 3)
        self.assertEqual(nn4.layers[0].output_size(), 5)
        self.assertEqual(nn4.layers[1].input_size(), 5)
        self.assertEqual(nn4.layers[1].output_size(), 6)
        self.assertEqual(nn4.layers[2].input_size(), 6)
        self.assertEqual(nn4.layers[2].output_size(), 3)

        self.assertTrue(isinstance(nn4.layers[0].activation_functions[0], ReLU))
        self.assertTrue(isinstance(nn4.layers[1].activation_functions[0], HardSigmoid))
        self.assertTrue(isinstance(nn4.layers[-1].activation_functions[0], Identity))

    def test_from_tf_model_5(self):
        dummy_input = tf.random.uniform((1, 3))
        TF_MODEL_5(dummy_input) # Build the model

        nn5 = NeuralNetwork.from_model(TF_MODEL_5)
        self.assertEqual(nn5.size(), 3)
        self.assertEqual(nn5.layers[0].input_size(), 3)
        self.assertEqual(nn5.layers[0].output_size(), 5)
        self.assertEqual(nn5.layers[1].input_size(), 5)
        self.assertEqual(nn5.layers[1].output_size(), 6)
        self.assertEqual(nn5.layers[2].input_size(), 6)
        self.assertEqual(nn5.layers[2].output_size(), 3)

        self.assertTrue(isinstance(nn5.layers[0].activation_functions[0], ReLU))
        self.assertTrue(isinstance(nn5.layers[1].activation_functions[0], HardSigmoid))
        self.assertTrue(isinstance(nn5.layers[2].activation_functions[0], Identity))