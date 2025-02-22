from NeuralNetworks import NeuralNetwork, to_pfsp_probabilistic
from ApproxBisimulation import ApproxBisV1PFSPManager, ProbabilisticFiniteStateProcess

import torch
import tensorflow as tf
import math

SEED = 1234
NUMBER_OF_SAMPLES = 100000

def compute_approximate_bisimulation(model1: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                                     model2: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                                     input_bounds: list[tuple[float, float] | list[tuple[float, float]]] | None = None,
                                     precision: float = 0.01,
                                     verbose: bool = False) -> tuple[float, tuple[float, float]]:

    pfsp1 = _convert_to_pfsp(model1, input_bounds)
    pfsp2 = _convert_to_pfsp(model2, input_bounds)
    manager = ApproxBisV1PFSPManager(pfsp1, pfsp2)
    lower = 0
    upper = 1

    for i in range(math.ceil(math.log(1 / precision, 2))):
        epsilon = (lower + upper) / 2
        are_bisimilar = manager.evaluate_probabilistic_approximate_bisimulation(epsilon, verbose)

        if are_bisimilar:
            upper = epsilon
        else:
            lower = epsilon

    return (lower + upper) / 2, (lower, upper)


def _convert_to_pfsp(model: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                    input_bounds: list[tuple[float, float] | list[tuple[float, float]]] | None = None
                     ) -> ProbabilisticFiniteStateProcess:

    if not isinstance(model, NeuralNetwork):
        model = NeuralNetwork.from_model(model)

    return to_pfsp_probabilistic(model=model,
                                 input_bounds=input_bounds,
                                 number_of_samples=NUMBER_OF_SAMPLES,
                                 seed=SEED)