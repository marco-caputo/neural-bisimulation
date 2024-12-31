from z3 import *
import torch
from NNToGraph import TensorFlowFFNN, TorchFFNN


def main():
    model = TorchFFNN([2, 3, 4, 1])
    for layer_idx, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            # Affine transformation: z_j = sum_k (x_k * W_kj) + b_j
            weights = layer.weight.detach().numpy()
            biases = layer.bias.detach().numpy()
            print(weights)
            print(biases)


if __name__ == "__main__":
    main()