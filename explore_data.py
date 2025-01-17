import pandas as pd

# Load the parsed dataset
try:
    df = pd.read_csv('parsed_cars_data.csv')
except FileNotFoundError:
    print("Error: parsed_cars_data.csv not found. Please ensure the file is in the correct directory.")
    exit()

# Explore the dataset
print("Head of the dataset:")
print(df.head())

print("\\nInfo of the dataset:")
print(df.info())

print("\\nDescription of the dataset:")
print(df.describe())

# Analyze unique values in categorical columns
print("\\nUnique values in categorical columns:")
for column in df.select_dtypes(include='object').columns:
    if column == 'Engines':
        continue
    if column == 'Turbo':
        print(f"Column '{column}': {df[column].nunique()} unique values")
        print(f"Unique values in '{column}':\\n{df[column].unique().tolist()}")
        # Count the occurrences of each value
        value_counts = df[column].value_counts()
        print(f"\\nCounts of each value in '{column}':\\n{value_counts}")