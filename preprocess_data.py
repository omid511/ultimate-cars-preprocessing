import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

def preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)

    # Treat 'Seats' as categorical
    df['Seats'] = df['Seats'].astype('object')

    # Address NaN values using column mode
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

   # Separate target variable before outlier removal
    if 'Cars Prices' in df.columns:
        target = df['Cars Prices']
        df = df.drop('Cars Prices', axis=1)
    else:
        target = None

    # Encode categorical values
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

   # Remove outliers using IQR method, excluding target variable
    numerical_cols = df.select_dtypes(include=['number']).columns
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # Add target variable back if it was removed
    if target is not None:
        df['Cars Prices'] = target

    # Standardize numerical values
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Reduce numerical features using PCA
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    numerical_data = df[numerical_cols]
    reduced_data = pca.fit_transform(numerical_data)
    reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)])

    # Combine reduced numerical features with categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    final_df = pd.concat([reduced_df, df[categorical_cols]], axis=1)

   # Split into train and test sets, handling missing target variable
    if 'Cars Prices' in final_df.columns:
        X = final_df.drop('Cars Prices', axis=1)
        y = final_df['Cars Prices']
    else:
        # If 'Cars Prices' is missing, use a different column or strategy
        # For example, use the first column as a placeholder target
        X = final_df.iloc[:, 1:]
        y = final_df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Example usage
X_train, X_test, y_train, y_test = preprocess_data('parsed_cars_data.csv')
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)