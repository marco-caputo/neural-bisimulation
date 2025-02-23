import numpy as np
import pandas as pd

import time
from ApproxBisimulation import compute_approximate_bisimulation
from NeuralNetworks.ActivationFunctions import HardSigmoid

from SMTEquivalence.SMTEquivalenceComputer import *

# Lunch with with command line from root directory: python -m NNTraining.experiments

def special_trial():
    models = ["female_dim3_3",
              "female_dim3_5",
              "female_dim4_3",
              "male_dim3_3",
              "male_dim3_5",
              "male_dim4_3",
              "diet_dim3_3",
              "diet_dim3_5",
              "diet_dim4_3"
              ]

    time_matrix = np.full((1, len(models)), np.nan)
    for i, model in enumerate(models):
        print(f"Comparing models {model} and {model}")
        model1 = tf.keras.models.load_model(f"NNTraining/models/model_{model}_out1.keras")
        model2 = tf.keras.models.load_model(f"NNTraining/models/model_{model}_out1.keras")
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
        time_matrix[0, i] = time_taken

    df = pd.DataFrame(time_matrix, columns=models)
    df.to_csv(f"NNTraining/results/smt_special_1.csv", index=False)

    print("Files saved successfully!")

def approximate_equivalence_trial_1():
    # List of object names
    models = ["female_dim3_3",
              "female_dim3_5",
              "female_dim4_3",
              "male_dim3_3",
              "male_dim3_5",
              "male_dim4_3",
              "diet_dim3_3",
              "diet_dim3_5",
              "diet_dim4_3"
              ]

    size = len(models)
    similarity_matrix = np.full((size, size), np.nan)
    ranges_matrix = np.full((size, size), np.nan, dtype=object)
    counterexamples_matrix = np.full((size, size), np.nan, dtype=object)
    time_matrix = np.full((size, size), np.nan)

    for i in range(size):
        for j in range(i+1):
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

    for matrix, default_value, name in [(similarity_matrix, "metric"),
                                        (ranges_matrix, "ranges"),
                                        (counterexamples_matrix, "counterexamples"),
                                        (time_matrix, "time")]:
        df = pd.DataFrame(matrix, columns=models)
        df.insert(0, "Model", models)

        if name in ["ranges", "counterexamples"]:
            df.to_pickle(f"NNTraining/results/smt_{name}_1.pkl")
        else:
            df.to_csv(f"NNTraining/results/smt_{name}_1.csv", index=False)

    print("Files saved successfully!")


def approximate_equivalence_trial_2():
    for dim in ["dim3_3",
                "dim3_5",
                "dim4_3"
                ]:

        models = [f"heart_{dim}",
                  f"heart_{dim}",
                  f"depression_{dim}",
                  f"heart_inverse_{dim}"
                  ]

        size = len(models)
        similarity_matrix = np.full((size, size), np.nan)
        ranges_matrix = np.full((size, size), np.nan, dtype=object)
        counterexamples_matrix = np.full((size, size), np.nan, dtype=object)
        time_matrix = np.full((size, size), np.nan)

        for i in range(size):
            for j in range(i+1):
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

        for matrix, name in [(similarity_matrix, "metric"),
                             (ranges_matrix, "ranges"),
                             (counterexamples_matrix, "counterexamples"),
                             (time_matrix, "time")]:
            df = pd.DataFrame(matrix, columns=models)
            df.insert(0, "Model", models)

            if name in ["ranges", "counterexamples"]:
                df.to_pickle(f"NNTraining/results/smt_{name}_{dim}_2.pkl")
            else:
                df.to_csv(f"NNTraining/results/smt_{name}_{dim}_2.csv", index=False)

        print("Files saved successfully!")


def approximate_bisimulation_trial_1():
    # List of object names
    models = ["female_dim3_3",
              "female_dim3_5",
              "female_dim4_3",
              "male_dim3_3",
              "male_dim3_5",
              "male_dim4_3",
              "diet_dim3_3",
              "diet_dim3_5",
              "diet_dim4_3"
              ]

    for max_avg in [True, False]:
        size = len(models)
        similarity_matrix = np.full((size, size), np.nan)
        time_matrix = np.full((size, size), np.nan)

        for i in range(0, size):
            for j in range(i+1):
                print(f"Comparing models {models[i]} and {models[j]}")
                model1 = tf.keras.models.load_model(f"NNTraining/models/model_{models[i]}_out2.keras")
                model2 = tf.keras.models.load_model(f"NNTraining/models/model_{models[j]}_out2.keras")
                nn1 = NeuralNetwork.from_model(model1)
                nn2 = NeuralNetwork.from_model(model2)

                start_time = time.time()
                similarity = compute_approximate_bisimulation(
                    nn1, nn2, input_bounds=[(-3, 3)] * 3 + [[(0, 0), (1, 1)]], max_avg=max_avg
                )
                end_time = time.time()
                time_taken = round(end_time - start_time, 3)
                similarity = round(similarity, 3)
                print(f"Similarity: {similarity}, Time: {time_taken}")

                for matrix, value in [(similarity_matrix, similarity),(time_matrix, time_taken)]:
                    matrix[i, j] = value
                    matrix[j, i] = value

        for matrix, name in [(similarity_matrix, "metric"), (time_matrix, "time")]:
            df = pd.DataFrame(matrix, columns=models)
            df.insert(0, "Model", models)
            df.to_csv(f"NNTraining/results/bis_{'max' if max_avg else 'avg'}_{name}_1.csv", index=False)

    print("Files saved successfully!")


def approximate_bisimulation_trial_2():
    dims = ["dim3_3", "dim3_5", "dim4_3"]
    models = ["heart", "diet", "depression", "heart_inverse"]

    for max_avg in [True, False]:
        similarity_matrix = np.full((len(dims), 4), np.nan)
        time_matrix = np.full((len(dims), 4), np.nan)

        for i, dim in enumerate(dims):
            for j, model in enumerate(models):
                print(f"Comparing models {models[0]}_{dim} and {model}_{dim}")
                model1 = tf.keras.models.load_model(f"NNTraining/models/model_{models[0]}_{dim}_out2.keras")
                model2 = tf.keras.models.load_model(f"NNTraining/models/model_{model}_{dim}_out2.keras")
                nn1 = NeuralNetwork.from_model(model1)
                nn2 = NeuralNetwork.from_model(model2)

                start_time = time.time()
                similarity = compute_approximate_bisimulation(
                    nn1, nn2, input_bounds=[(-3, 3)] * 3 + [[(0, 0), (1, 1)]], max_avg=max_avg
                )
                end_time = time.time()
                time_taken = round(end_time - start_time, 3)
                similarity = round(similarity, 3)
                print(f"Similarity: {similarity}, Time: {time_taken}")

                for matrix, value in [(similarity_matrix, similarity), (time_matrix, time_taken)]:
                    matrix[i, j] = value

        for matrix, name in [(similarity_matrix, "metric"), (time_matrix, "time")]:
            df = pd.DataFrame(matrix, columns=models)
            df.insert(0, "Dimension", dims)
            df.to_csv(f"NNTraining/results/bis_{'max' if max_avg else 'avg'}_{name}_2.csv", index=False)

    print("Files saved successfully!")


if __name__ == "__main__":
    special_trial()
