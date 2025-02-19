from NNTraining.neural_network import *


def train_networks():
    # Transforming the Datasets (Comment Out if Already Done)
    for dataset in ["datasets/female_dataset", "datasets/male_dataset", "datasets/female_diet_dataset"]:
        divide_dataset(dataset)

    for dataset, label in [("male", "Heart_Attack_Outcome"), ("female", "Heart_Attack_Outcome"), ("female_diet", "Diet_Quality")]:
        for layers, hidden_neurons in [(1, 4), (2, 5), (4, 8)]:
            for output_neurons in [1, 2]:
                small_nn_males = StrokeNeuralNetwork(layers, hidden_neurons, 5, output_neurons)
                small_nn_males.train(dataset, label, ["Smoking_Status"])


def train_and_save_models():
    # Transforming the Datasets (Comment Out if Already Done)
    divide_dataset()
    # Train and save networks
    train_networks()


if __name__ == "__main__":
    train_and_save_models()
