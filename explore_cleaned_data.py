import pandas as pd

# Load the parsed dataset
df = pd.read_csv("preprocessed_cars_data.csv")


# Explore the dataset
print("Head of the dataset:")
print(df.head())

print("\nInfo of the dataset:")
print(df.info())

print("\nDescription of the dataset:")
print(df.describe())

# Check for NaN values
print("\nColumns with NaN values:")
nan_columns = df.columns[df.isnull().any()].tolist()
if nan_columns:
    for col in nan_columns:
        nan_count = df[col].isnull().sum()
        print(f"  {col}: {nan_count} NaN values")
else:
    print("None")
