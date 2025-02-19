import numpy as np
import pandas as pd
import tensorflow as tf
import time
from NeuralNetworks import NeuralNetwork, HardSigmoid

from SMTEquivalence.SMTEquivalenceComputer import compute_approximate_equivalence_experiment

def approximate_equivalence_trial():
    # List of object names
    models = ["female_dim3_4",
              "female_dim3_7",
              "female_dim4_7",
              "male_dim3_4",
              "male_dim3_7",
              "male_dim4_7",
              "diet_dim3_4",
              "diet_dim3_7",
              "diet_dim4_7"
              ]

    size = len(models)
    similarity_matrix = np.full((size, size), np.nan)
    ranges_matrix = np.full((size, size), np.nan, dtype=object)
    counterexamples_matrix = np.full((size, size), np.nan, dtype=object)
    time_matrix = np.full((size, size), np.nan)

    for i in range(1, size):
        for j in range(i):
            print(f"Comparing models {models[i]} and {models[j]}")
            model1 = tf.keras.models.load_model(f"NNTraining/models/model_{models[i]}_out1.keras")
            model2 = tf.keras.models.load_model(f"NNTraining/models/model_{models[j]}_out1.keras")
            nn1 = NeuralNetwork.from_model(model1)
            nn2 = NeuralNetwork.from_model(model2)
            nn1.layers[-1].activation_functions = [HardSigmoid()] * nn1.layers[-1].output_size()
            nn2.layers[-1].activation_functions = [HardSigmoid()] * nn2.layers[-1].output_size()

            start_time = time.time()
            similarity, similarity_range, counterexamples = compute_approximate_equivalence_experiment(
                nn1, nn2, input_bounds=[(-2, 2)] * 3 + [[(0, 0), (1, 1)]], p=1, precision=0.01, verbose=True
            )
            end_time = time.time()
            time_taken = int(end_time - start_time)

            print(f"Similarity: {similarity}, Similarity Range: {similarity_range}, time: {time_taken}")

            for matrix, value in [(similarity_matrix, similarity), (ranges_matrix, similarity_range),
                                     (counterexamples_matrix, counterexamples), (time_matrix, time_taken)]:
                matrix[i, j] = value
                matrix[j, i] = value

    for matrix, default_value, name in [(similarity_matrix, 0.00390625, "metric"),
                                        (ranges_matrix, [0, 0.0078125], "ranges"),
                                        (counterexamples_matrix, [], "counterexamples"),
                                        (time_matrix, np.nan, "time")]:
        np.fill_diagonal(matrix, default_value)
        df = pd.DataFrame(matrix, columns=models)
        df.insert(0, "Model", models)

        if name in ["ranges", "counterexamples"]:
            df.to_pickle(f"NNTraining/results/{name}.pkl")
        else:
            df.to_csv(f"NNTraining/results/{name}.csv", index=False)

    print("Files saved successfully!")


if __name__ == "__main__":
    approximate_equivalence_trial()