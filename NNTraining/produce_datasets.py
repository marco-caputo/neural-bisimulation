import pandas as pd


def remove_outliers(df, columns):
    # Calculate outlier masks for all specified columns
    outlier_mask = pd.Series(False, index=df.index)  # Start with no outliers

    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = abs(Q3 - Q1)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Mark rows that are outliers
        outlier_mask |= (df[column] < lower_bound) | (df[column] > upper_bound)

    # Count and print number of discarded rows
    removed_rows = outlier_mask.sum()
    print(f"Total rows discarded due to outliers: {removed_rows}")

    # Remove all identified outliers in one step
    return df[~outlier_mask]


def preprocess_datasets():
    # Load the dataset
    dataset_path = 'datasets/heart_attack_dataset.csv'
    df = pd.read_csv(dataset_path)

    # Preprocessing
    df["Smoking_Status"] = df["Smoking_Status"].map({"No": 0, "Yes": 1})
    df["Diet_Quality"] = df["Diet_Quality"].map({"Poor": 1, "Good": 0, "Average": 2})

    # Remove outliers across all target columns at once
    df_cleaned = remove_outliers(df, ['Cholesterol_Level', 'Blood_Pressure_Systolic', 'Obesity_Index'])

    # Filter 1: Remove females and keep specific columns for analysis
    df_females_removed = df[df['Gender'] != 'Female'][['Cholesterol_Level',
                                                       'Blood_Pressure_Systolic',
                                                       'Obesity_Index',
                                                       'Smoking_Status',
                                                       'Heart_Attack_Outcome']]

    # Filter 2: Remove males and keep the same columns as Filter 1
    df_males_removed = df[df['Gender'] != 'Male'][['Cholesterol_Level',
                                                   'Blood_Pressure_Systolic',
                                                   'Obesity_Index',
                                                   'Smoking_Status',
                                                   'Heart_Attack_Outcome']]

    # Filter 3: Remove Average diet quality and keep Diet_Quality instead of Heart_Attack_Outcome
    df_males_removed_diet = df[df['Diet_Quality'] != 2][['Cholesterol_Level',
                                                         'Blood_Pressure_Systolic',
                                                         'Obesity_Index',
                                                         'Smoking_Status',
                                                         'Diet_Quality']]

    # Save the filtered datasets to new CSV files
    df_females_removed.to_csv('datasets/male_dataset.csv', index=False)
    df_males_removed.to_csv('datasets/female_dataset.csv', index=False)
    df_males_removed_diet.to_csv('datasets/diet_dataset.csv', index=False)
