import pandas as pd


def divide_dataset():
    # Load the dataset
    dataset_path = 'datasets/heart_attack_dataset.csv'
    df = pd.read_csv(dataset_path)

    # Preprocessing
    df["Smoking_Status"] = df["Smoking_Status"].map({"No": 0, "Yes": 1})
    df["Diet_Quality"] = df["Diet_Quality"].map({"Poor": 0, "Good": 1, "Average": 2})

    # Filter 1: Remove females and keep specific columns for analysis
    df_females_removed = df[df['Gender'] != 'Female'][['Age',
                                                       'Cholesterol_Level',
                                                       'Blood_Pressure_Systolic',
                                                       'Obesity_Index',
                                                       'Smoking_Status',
                                                       'Heart_Attack_Outcome']]

    # Filter 2: Remove males and keep the same columns as Filter 1
    df_males_removed = df[df['Gender'] != 'Male'][['Age',
                                                   'Cholesterol_Level',
                                                   'Blood_Pressure_Systolic',
                                                   'Obesity_Index',
                                                   'Smoking_Status',
                                                   'Heart_Attack_Outcome']]

    # Filter 3: Remove Average diet quality and keep Diet_Quality instead of Heart_Attack_Outcome
    df_males_removed_diet = df[df['Diet_Quality'] != 2][['Age',
                                                        'Cholesterol_Level',
                                                        'Blood_Pressure_Systolic',
                                                        'Obesity_Index',
                                                        'Smoking_Status',
                                                        'Diet_Quality']]

    # Save the filtered datasets to new CSV files
    df_females_removed.to_csv('datasets/male_dataset.csv', index=False)
    df_males_removed.to_csv('datasets/female_dataset.csv', index=False)
    df_males_removed_diet.to_csv('datasets/female_diet_dataset.csv', index=False)


divide_dataset()