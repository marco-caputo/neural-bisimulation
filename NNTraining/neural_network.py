from sklearn.model_selection import train_test_split
import pandas as pd
from keras._tf_keras.keras.layers import Dense
from keras import losses, Sequential
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib


def divide_dataset(dataset_name:str):
    #Dividing the Datasets into Training and Testing Datasets
    df = pd.read_csv(f"{dataset_name}.csv")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(f"transformed_datasets/train_{dataset_name.replace('datasets/', '')}.csv", index=False)
    test_df.to_csv(f"transformed_datasets/test_{dataset_name.replace('datasets/', '')}.csv", index=False)


class NeuralNetwork:

    # Building a Neural Network Model with the Same Number of Neurons for Each Layer
    def __init__(self, layers: int, neurons: int, input_neurons, output_neurons=1):
        # Base Model with only 1 Layer
        model = Sequential()
        model.add(Dense(neurons, activation="relu", input_shape=(input_neurons,)))

        # Adding Eventual More Layers
        for _ in range(layers):
            model.add(Dense(neurons, activation="relu"))

        #activation = "softmax" if results > 2 else "sigmoid"
        loss = (
            losses.SparseCategoricalCrossentropy(from_logits=True) if output_neurons > 2 else (  # Label is [0, 1, 2, ...]
            losses.CategoricalCrossentropy(from_logits=True) if output_neurons == 2 else  # Label is [0, 1] or [1, 0]
            losses.BinaryCrossentropy(from_logits=True))  # Label is 0 or 1
        )

        model.add(Dense(output_neurons))
        model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

        self.model = model

    def train(self, dataset_name: str, label: str, non_continuous_columns: list[str] = None):
        train_df = pd.read_csv(f"transformed_datasets/train_{dataset_name}_dataset.csv")
        test_df = pd.read_csv(f"transformed_datasets/test_{dataset_name}_dataset.csv")

        # Identify continuous features
        continuous_columns = [col for col in train_df.columns if col not in non_continuous_columns + [label]]

        # Scaling the Data
        sc = StandardScaler()
        sc.fit(train_df[continuous_columns])

        train_df[continuous_columns] = sc.transform(train_df[continuous_columns])
        test_df[continuous_columns] = sc.transform(test_df[continuous_columns])

        labels_train = train_df[f"{label}"]
        features_train = train_df.drop(columns=[f"{label}"])
        labels_test = test_df[f"{label}"]
        features_test = test_df.drop(columns=[f"{label}"])

        # Hot encoding the labels for binary classification with 2 output neurons
        if self.model.output_shape[-1] == 2 and labels_train.ndim == 1:
            encoder = OneHotEncoder(sparse_output=False)
            labels_train = encoder.fit_transform(labels_train.values.reshape(-1, 1))
            labels_test = encoder.transform(labels_test.values.reshape(-1, 1))

        # Training the Model
        self.model.fit(features_train, labels_train, epochs=30, batch_size=128, validation_data=(features_test, labels_test))

        # Save the trained model and scaler
        self.model.save(f"models/model_{dataset_name}_{self.model.layers[1].units}.h5")  # Save model
        joblib.dump(sc, f"models/scaler_{dataset_name}_{self.model.layers[1].units}.pkl")  # Save scaler