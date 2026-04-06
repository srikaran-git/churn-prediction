# src/data/preprocessor.py
"""
Data Preprocessor module for the Churn Prediction pipeline.

This module will handle all data cleaning and transformation steps required
before model training.
"""

# Standard Library
from typing import List, Tuple

# Third-party Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Local Modules
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
columns_to_drop = ["customerID"]
categorical_columns = ["gender", "Partner", "Dependents", "PhoneService"]
target_column = "Churn"


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw customer data from a CSV file.

    Args:
        file_path: Path to the raw CSV data file.

    Returns:
        Raw DataFrame as loaded from disk.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        ValueError: If the file is empty.

    Example:
        >>> df = load_raw_data("data/raw/churn.csv")
        >>> print(df.shape)
        (7043, 21)
    """

    from pathlib import Path

    path = Path(file_path)
    if not path.is_file():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        logger.error(f"File is empty: {file_path}")
        raise ValueError(f"File is empty: {file_path}")

    logger.info(
        f"Loaded raw data from {file_path} with shape {df.shape}"
        f" columns: {df.columns.tolist()}"
    )
    return df


def remove_irrelevant_columns(
    df: pd.DataFrame, columns: List[str] = columns_to_drop
) -> pd.DataFrame:
    """
    Remove irrelevant columns from the DataFrame.

    Args:
        df: Input DataFrame.
        columns: List of column names to drop.
    Returns:
        DataFrame with specified columns removed.
    """
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    dropped_df = df.drop(columns=existing_columns)
    logger.info(f"Dropped columns: {existing_columns}")
    if not existing_columns:
        logger.warning(f"Columns {columns_to_drop} not found in DataFrame.")
    return dropped_df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with any missing values.

    In production, you might impute instead of dropping.
    This simple strategy is a starting point.

    Args:
        df: Input DataFrame, possibly with NaN values.

    Returns:
        DataFrame with no missing values.
    """
    initial_shape = df.shape
    df_clean = df.dropna()
    logger.info(
        f"Dropped {initial_shape[0] - df_clean.shape[0]} rows with missing values."
    )
    return df_clean


def encode_categorical_columns(
    df: pd.DataFrame, columns: List[str] = categorical_columns
) -> pd.DataFrame:
    """
    Encode categorical variables using Label Encoding.

    Args:
        df: Input DataFrame with categorical columns.
        columns: List of column names to encode.

    Returns:
        DataFrame with specified categorical columns encoded as integers.

    Note:
    This modifies a copy of the DataFrame, never the original.
        For production, consider OrdinalEncoder or OneHotEncoder
        inside an sklearn Pipeline (covered on Day 10).
    """
    df_encoded = df.copy()
    encoder = LabelEncoder()

    for column in categorical_columns:
        if column not in df_encoded.columns:
            logger.warning(f"Column '{column}' not found — skipping")
            continue

        df_encoded[column] = encoder.fit_transform(df_encoded[column])
        logger.info(f"Encoded column: '{column}'")

    return df_encoded


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = target_column,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate the DataFrame into feature matrix X and target vector y.

    Args:
        df: Full DataFrame including target column.
        target_column: Name of the column to predict.

    Returns:
        Tuple of (X, y) where X is features and y is target.

    Raises:
        ValueError: If target_column is not in the DataFrame.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    logger.info(
        f"Split data into features X with shape {X.shape} "
        f"a target y with shape {y.value_counts(normalize=True)}"
    )
    return X, y


def preprocess_pipeline(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Run the full preprocessing pipeline end-to-end.

    This is the main function to call from train.py.
    It chains all preprocessing steps in the correct order.

    Args:
        file_path: Path to the raw CSV data file.

    Returns:
        Tuple of (X, y) ready for model training.

    Example:
        >>> X, y = preprocess_pipeline("data/raw/churn.csv")
        >>> print(X.shape, y.shape)
    """
    logger.info("Starting preprocessing pipeline...")

    # Step 1: Load
    df = load_raw_data(file_path)

    # Step 2: Drop useless columns
    df = remove_irrelevant_columns(df)

    # Step 3: Handle missing values
    df = handle_missing_values(df)

    # Step 4: Encode categoricals
    df = encode_categorical_columns(df)

    # Step 5: Split X and y
    X, y = split_features_and_target(df)

    logger.info(f"Preprocessing complete — " f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y
