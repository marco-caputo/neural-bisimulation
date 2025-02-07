from z3 import *
from NeuralNetworks import *


def main():
    x = Real("x")
    print(simplify(And(0 < x, x < 0)))

    model = TorchFFNN([2, 3, 4, 1])

    print(input_dim(model))
    print(output_dim(model))


if __name__ == "__main__":
    main()