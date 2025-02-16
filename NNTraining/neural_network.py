from sklearn.model_selection import train_test_split
import pandas as pd
from keras._tf_keras.keras.layers import Dense
from keras import Sequential
from sklearn.preprocessing import StandardScaler


def divide_dataset(dataset_name:str):
    #Dividing the Datasets into Training and Testing Datasets
    df = pd.read_csv(f"{dataset_name}.csv", sep=";")

    df["Smoking_Status"] = df["Smoking_Status"].map({"No": 0, "Yes": 1})
    if "Stress_Level" in df.columns:
        df["Stress_Level"] = df["Stress_Level"].map({"Low": 0, "Medium": 1, "High": 2})

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(f"transformed_datasets/train_{dataset_name.replace('datasets/', '')}.csv", index=False)
    test_df.to_csv(f"transformed_datasets/test_{dataset_name.replace('datasets/', '')}.csv", index=False)


class NeuralNetwork:

    # Building a Neural Network Model with the Same Number of Neurons for Each Layer
    def __init__(self, layers: int, neurons: int, train_df, results=1):
        # Base Model with only 1 Layer
        model = Sequential()
        model.add(Dense(neurons, activation="relu", input_shape=(train_df.shape[1]-1,)))

        # Adding Eventual More Layers
        for _ in range(layers - 1):
            model.add(Dense(neurons, activation="relu"))

        #activation = "softmax" if results > 2 else "sigmoid"
        loss = "sparse_categorical_crossentropy" if results > 2 else "binary_crossentropy"

        model.add(Dense(results))
        model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

        self.model = model

    def train(self, dataset_name:str, label:str):
        train_df = pd.read_csv(f"transformed_datasets/train_{dataset_name}_dataset.csv")
        test_df = pd.read_csv(f"transformed_datasets/test_{dataset_name}_dataset.csv")

        # Scaling the Data
        sc = StandardScaler()

        labels_train = train_df[f"{label}"]
        features_train = sc.fit_transform(train_df.drop(columns=[f"{label}"]))

        labels_test = test_df[f"{label}"]
        features_test = sc.transform(test_df.drop(columns=[f"{label}"]))


        # Training the Model
        self.model.fit(features_train, labels_train, epochs=30, batch_size=128, validation_data=(features_test, labels_test))