import tensorflow as tf

from NeuralNetworks import NeuralNetwork, HardSigmoid
from SMTEquivalence import compute_approximate_equivalence, are_approximate_equivalent, encode_into_SMT_formula


def approximate_equivalence():
    model1 = tf.keras.models.load_model("models/model_male_dim3_out1.keras")
    model2 = tf.keras.models.load_model("models/model_female_dim3_out1.keras")
    nn1 = NeuralNetwork.from_model(model1)
    nn2 = NeuralNetwork.from_model(model2)
    nn1.layers[-1].activation_functions = [HardSigmoid()]*nn1.layers[-1].output_size()
    nn2.layers[-1].activation_functions = [HardSigmoid()]*nn2.layers[-1].output_size()
    distance = compute_approximate_equivalence(
            nn1, nn2, input_bounds=[(-2, 2)]*4 + [[(0, 0), (1, 1)]], p=1, precision=0.01, verbose=True
        )

    print(f"Final result = {distance}")


if __name__ == "__main__":
    approximate_equivalence()