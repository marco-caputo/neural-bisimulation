from NNTraining import *
from sklearn.model_selection import train_test_split


def divide_dataset(dataset_name: str):
    #Dividing the Datasets into Training and Testing Datasets
    df = pd.read_csv(f"{dataset_name}.csv")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(f"transformed_datasets/train_{dataset_name.replace('datasets/', '')}.csv", index=False)
    test_df.to_csv(f"transformed_datasets/test_{dataset_name.replace('datasets/', '')}.csv", index=False)


def train_networks_heart():
    # Transforming the Datasets (Comment Out if Already Done)
    for dataset in ["datasets/female_dataset", "datasets/male_dataset", "datasets/diet_dataset"]:
        divide_dataset(dataset)

    for dataset, label in [("male", "Heart_Attack_Outcome"), ("female", "Heart_Attack_Outcome"),
                           ("diet", "Diet_Quality")]:
        for layers, hidden_neurons in [(1, 3), (1, 5), (2, 3)]:
            for output_neurons in [1, 2]:
                model = StrokeNeuralNetwork(layers, hidden_neurons, 4, output_neurons)
                model.train(dataset, label, ["Smoking_Status"])


def train_networks_parkinson():
    divide_dataset("datasets/parkinson_dataset")

    for layers, hidden_neurons in [(1, 3), (1, 5), (2, 3)]:
        for output_neurons in [1, 2]:
            model = StrokeNeuralNetwork(layers, hidden_neurons, 4, output_neurons)
            model.train("parkinson", "Diagnosis", ["Smoking"])

def train_networks_depression():
    divide_dataset("datasets/depression_dataset")

    for layers, hidden_neurons in [(1, 3), (1, 5), (2, 3)]:
        for output_neurons in [1, 2]:
            model = StrokeNeuralNetwork(layers, hidden_neurons, 4, output_neurons)
            model.train("depression", "Depression", ["Gender"])


def train_networks_heart_inverse():
    divide_dataset("datasets/heart_dataset")
    divide_dataset("datasets/heart_inverse_dataset")

    for dataset, label in [("heart", "Heart_Attack_Outcome"), ("heart_inverse", "Heart_Attack_Outcome")]:
        for layers, hidden_neurons in [(1, 3), (1, 5), (2, 3)]:
            for output_neurons in [1, 2]:
                model = StrokeNeuralNetwork(layers, hidden_neurons, 4, output_neurons)
                model.train(dataset, label, ["Smoking_Status"])


def train_and_save_models():
    # Transforming the Datasets (Comment Out if Already Done)
    #preprocess_datasets_heart()
    preprocess_datasets_parkinson()
    preprocess_datasets_depression()
    preprocess_datasets_heart_inverse()
    # Train and save networks
    #train_networks_heart()
    train_networks_parkinson()
    train_networks_depression()
    train_networks_heart_inverse()


if __name__ == "__main__":
    train_and_save_models()
