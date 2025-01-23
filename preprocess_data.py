import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer


def preprocess_data(filepath):
    # Load the dataset and keep track of original NaN indices in target
    df = pd.read_csv(filepath)
    original_nan_indices = df.index[df["Cars Prices"].isnull()]

    # Treat 'Seats' as categorical
    df["Seats"] = df["Seats"].astype("object")

    # Drop cars names column
    df = df.drop("Cars Names", axis=1)

    # Separate target variable before any preprocessing
    target = df["Cars Prices"]
    df = df.drop("Cars Prices", axis=1)

    # Address NaN values using KNN imputation
    numerical_cols_for_imputation = df.select_dtypes(include=["number"]).columns
    imputer = KNNImputer(n_neighbors=5)
    df[numerical_cols_for_imputation] = imputer.fit_transform(
        df[numerical_cols_for_imputation]
    )

    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Add target back to the dataframe for outlier detection
    if target is not None:
        df["Cars Prices"] = target

        # Reset 'Cars Prices' to NaN where it was originally NaN
        df.loc[original_nan_indices, "Cars Prices"] = pd.NA

    # Identify and remove outliers using IQR method, considering all columns
    outlier_mask = pd.Series(
        [False] * len(df), index=df.index
    )  # Initialize mask with False

    for col in df.columns:  # Iterate over all columns
        # Skip non-numeric and boolean columns
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(
            df[col]
        ):
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        col_outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_mask = (
            outlier_mask | col_outlier_mask
        )  # Update mask for rows with outliers in any column

    # Remove rows with outliers from both df and target
    if target is not None:
        # Only remove from target if not originally NaN
        target = target[~outlier_mask | original_nan_indices]
    df = df[~outlier_mask]
    print(f"Number of removed outliers: {outlier_mask.sum()}")

    # Separate target variable after outlier removal
    if "Cars Prices" in df.columns:
        df = df.drop("Cars Prices", axis=1)

    # Update numerical_cols to exclude target variable for PCA
    numerical_cols = df.select_dtypes(include=["number", "bool"]).columns

    # Standardize numerical values
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    print(f"Number of features before PCA: {len(df.columns)}")

    # Reduce numerical features using PCA, excluding target variable
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    numerical_data = df[numerical_cols]
    reduced_data = pca.fit_transform(numerical_data)
    reduced_df = pd.DataFrame(
        reduced_data, columns=[f"PC{i}" for i in range(1, pca.n_components_ + 1)]
    )

    print(f"Number of features after PCA: {pca.n_components_}")

    # Combine reduced numerical features with one-hot encoded categorical features
    # Ensure the index of categorical_data aligns with reduced_df before concatenation
    categorical_data = df.select_dtypes(include=["number"]).drop(numerical_cols, axis=1)
    categorical_data = categorical_data.reindex(reduced_df.index)  # Align index
    final_df = pd.concat([reduced_df, categorical_data], axis=1)

    # Create a new Series with the same index as final_df
    target_reindexed = pd.Series(index=final_df.index)

    # Find the common indices between target and final_df
    common_indices = target.index.intersection(final_df.index)

    # Populate the new Series with values from target for common indices
    target_reindexed.loc[common_indices] = target.loc[common_indices]

    # Assign the reindexed target to the 'Cars Prices' column of final_df
    final_df["Cars Prices"] = target_reindexed

    # Scale the target variable
    if target is not None:
        target_scaler = StandardScaler()
        final_df[["Cars Prices"]] = target_scaler.fit_transform(
            final_df[["Cars Prices"]]
        )

    return final_df


# Example usage
final_df = preprocess_data("parsed_cars_data.csv")
final_df.to_csv("preprocessed_cars_data.csv", index=False)

# Split into train and test sets
X = final_df.drop("Cars Prices", axis=1)
y = final_df["Cars Prices"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Save the train and test sets separately
# X_train.to_csv("X_train.csv", index=False)
# X_test.to_csv("X_test.csv", index=False)
# y_train.to_csv("y_train.csv", index=False)
# y_test.to_csv("y_test.csv", index=False)
