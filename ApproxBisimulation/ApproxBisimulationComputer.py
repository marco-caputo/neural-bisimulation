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
                                     max_avg: bool = True) -> float:
    """
    Computes the approximate bisimilarity distance between two neural networks.
    The two given models are converted to Probabilistic Finite State Processes (PFSP) though probabilistic sampling
    according to the given input bounds. The PFSPs have a state for each node, each representing the state where the
    corresponding node has the maximum output value in its layer.
    The approximate bisimilarity distance is then computed between the two PFSPs according to max_avg flag,
    where max_avg=True computes the maximum distance among layers and max_avg=False computes the average
    distance among layers in the network.

    :param model1: the first neural network
    :param model2: the second neural network
    :param input_bounds: the input bounds for the neural networks
    :param max_avg: flag to compute the maximum or average distance among layers
    """
    pfsp1 = _convert_to_pfsp(model1, input_bounds)
    pfsp2 = _convert_to_pfsp(model2, input_bounds)
    return ApproxBisV1PFSPManager(pfsp1, pfsp2).evaluate_probabilistic_approximate_bisimilarity_max() if max_avg \
        else ApproxBisV1PFSPManager(pfsp1, pfsp2).evaluate_probabilistic_approximate_bisimilarity_avg()


def _convert_to_pfsp(model: NeuralNetwork | torch.nn.Module | tf.keras.Model,
                     input_bounds: list[tuple[float, float] | list[tuple[float, float]]] | None = None
                     ) -> ProbabilisticFiniteStateProcess:
    if not isinstance(model, NeuralNetwork):
        model = NeuralNetwork.from_model(model)
    return to_pfsp_probabilistic(model=model,
                                 input_bounds=input_bounds,
                                 number_of_samples=NUMBER_OF_SAMPLES,
                                 seed=SEED)
