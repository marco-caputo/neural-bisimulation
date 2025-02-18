from NNTraining.neural_network import *


def train_networks():
    # Transforming the Datasets (Comment Out if Already Done)
    divide_dataset("datasets/female_dataset")
    divide_dataset("datasets/male_dataset")
    divide_dataset("datasets/female_diet_dataset")

    # Training the Neural Networks
    small_nn_males = NeuralNetwork(1, 4, 5, 2)
    small_nn_males.train("male", "Heart_Attack_Outcome", ["Smoking_Status"])
    medium_nn_males = NeuralNetwork(2, 5, 5, 2)
    medium_nn_males.train("male", "Heart_Attack_Outcome", ["Smoking_Status"])
    large_nn_males = NeuralNetwork(4, 8, 5, 2)
    large_nn_males.train("male", "Heart_Attack_Outcome", ["Smoking_Status"])

    small_nn_females = NeuralNetwork(1, 4, 5, 2)
    small_nn_females.train("female", "Heart_Attack_Outcome", ["Smoking_Status"])
    medium_nn_females = NeuralNetwork(2, 5, 5, 2)
    medium_nn_females.train("female", "Heart_Attack_Outcome", ["Smoking_Status"])
    large_nn_females = NeuralNetwork(4, 8, 5, 2)
    large_nn_females.train("female", "Heart_Attack_Outcome", ["Smoking_Status"])

    small_nn_female_stress = NeuralNetwork(1, 4, 5, 2)
    small_nn_female_stress.train("female_diet", "Diet_Quality", ["Smoking_Status"])
    medium_nn_females = NeuralNetwork(2, 5, 5, 2)
    medium_nn_females.train("female_diet", "Diet_Quality", ["Smoking_Status"])
    large_nn_females = NeuralNetwork(4, 8, 5, 2)
    large_nn_females.train("female_diet", "Diet_Quality", ["Smoking_Status"])


train_networks()
