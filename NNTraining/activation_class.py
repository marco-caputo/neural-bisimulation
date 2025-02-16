import pandas as pd
from NNTraining.neural_network import *

# Transforming the Datasets (Comment Out if Already Done)
'''divide_dataset("datasets/female_dataset")
divide_dataset("datasets/male_dataset")
divide_dataset("datasets/female_stress_dataset")'''

# Training the Neural Networks
small_nn_males = NeuralNetwork(1, 4, pd.read_csv("transformed_datasets/train_male_dataset.csv"))
small_nn_males.train("male", "Heart_Attack_Outcome")
medium_nn_males = NeuralNetwork(2, 5, pd.read_csv("transformed_datasets/train_male_dataset.csv"))
medium_nn_males.train("male", "Heart_Attack_Outcome")
large_nn_males = NeuralNetwork(4, 8, pd.read_csv("transformed_datasets/train_male_dataset.csv"))
large_nn_males.train("male", "Heart_Attack_Outcome")

small_nn_females = NeuralNetwork(1, 4, pd.read_csv("transformed_datasets/train_female_dataset.csv"))
small_nn_females.train("female", "Heart_Attack_Outcome")
medium_nn_females = NeuralNetwork(2, 5, pd.read_csv("transformed_datasets/train_female_dataset.csv"))
medium_nn_females.train("female", "Heart_Attack_Outcome")
large_nn_females = NeuralNetwork(4, 8, pd.read_csv("transformed_datasets/train_female_dataset.csv"))
large_nn_females.train("female", "Heart_Attack_Outcome")

small_nn_female_stress = NeuralNetwork(1, 4, pd.read_csv("transformed_datasets/train_female_stress_dataset.csv"), 3)
small_nn_female_stress.train("female_stress", "Stress_Level")
medium_nn_females = NeuralNetwork(2, 5, pd.read_csv("transformed_datasets/train_female_stress_dataset.csv"))
medium_nn_females.train("female_stress", "Stress_Level")
large_nn_females = NeuralNetwork(4, 8, pd.read_csv("transformed_datasets/train_female_stress_dataset.csv"))
large_nn_females.train("female_stress", "Stress_Level")



