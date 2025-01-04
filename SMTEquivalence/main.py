from z3 import *
import time
import torch
from NNToGraph import TensorFlowFFNN, TorchFFNN


def main():
    z = Real('z')
    h = Real('h')
    l_params = {"max_val": 6, "threshold": 2, "negative_slope": 0.1}

    # Logical representation
    start = time.time()
    logical_form = simplify(Or(
        And(z >= l_params["max_val"], h == l_params["max_val"]),
        And(z >= l_params["threshold"], z < l_params["max_val"], h == z),
        And(z < l_params["threshold"], h == l_params["negative_slope"] * (z - l_params["threshold"]))
    ))
    end = time.time()
    print(f"Logical form time: {end - start}")

    # If-based representation
    start = time.time()
    if_form = simplify( h ==
        If(z >= l_params["max_val"],
           l_params["max_val"],
           If(z >= l_params["threshold"],
              z,
              l_params["negative_slope"] * (z - l_params["threshold"])
              )
           )
    )
    end = time.time()
    print(f"If-based form time: {end - start}")

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