from NeuralNetworks.ActivationFunctions import ActivationFunction, Identity


class Layer:
    weights: list[list[float]]
    biases: list[float]
    activation_functions: list[ActivationFunction]

    def __init__(self, weights: list[list[float]],
                 biases: list[float] = None,
                 activation_functions: list[ActivationFunction] = None):

        if not weights:
            raise ValueError("The weights tensor must not be empty.")
        if biases is None:
            biases = [0] * len(weights[0])
        if activation_functions is None:
            activation_functions = [Identity()] * len(weights[0])
        if len(weights[0]) != len(biases) or len(weights[0]) != len(activation_functions):
            raise ValueError("The dimension of weight tensor, biases and activation functions in the same layer must"
                             " be the coherent.")

        self.weights = weights
        self.biases = biases
        self.activation_functions = activation_functions


    def __deepcopy__(self, memodict={}):
        return Layer(self.weights.copy(),
                     self.biases.copy(),
                     [activation_function.copy() for activation_function in self.activation])

    def size(self):
        """
        Returns the number of neurons in the layer.
        This is equivalent to the output size of the layer.

        :return: the number of neurons
        """
        return len(self.weights[0])

    def output_size(self):
        """
        Returns the output size of the layer.
        This is equivalent to the number of neurons in the layer.

        :return: the output size
        """
        return self.size()

    def input_size(self):
        """
        Returns the number of neurons in the previous layer.

        :return: the number of neurons in the previous layer
        """
        return len(self.weights)

    def forward_pass(self, inputs: list[float]) -> list[float]:
        """
        Computes the output of the layer given the input.

        :param inputs: the input to the layer
        :return: the output of the layer
        """
        if len(inputs) != self.input_size():
            raise ValueError("The number of inputs must be equal to the number of neurons in the previous layer.")

        return [self.activation_functions[j](
                    sum(self.weights[i][j] * inputs[i] for i in range(len(inputs))) + self.biases[j]
                )
                for j in range(self.size())]

    def __len__(self):
        return self.size()

    def __call__(self, inputs: list[float]) -> list[float]:
        return self.forward_pass(inputs)

    def __repr__(self):
        return f"Weights: {self.weights}, Biases: {self.biases}, Activation Functions: {self.activation_functions}"