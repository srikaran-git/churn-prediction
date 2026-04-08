# src/data/preprocessor.py
"""
Data Preprocessor module for the Churn Prediction pipeline.

Responsibility: Clean and transform raw data into
model-ready features. Nothing else.
"""

# Standard Library
from typing import List, Tuple

# Third-party Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Local Modules
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
columns_to_drop = ["customerID"]

binary_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

target_column = "Churn"

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def drop_useless_columns(
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
    cols_to_drop = [col for col in columns if col in df.columns]
    result = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped columns: {cols_to_drop}")
    return result


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'TotalCharges' to numeric, coercing errors to NaN.
    """

    if "TotalCharges" not in df.columns:
        logger.warning("'TotalCharges' column not found — skipping conversion.")
        return df

    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    converted_nulls = df["TotalCharges"].isna().sum()

    logger.info(
        f"TotalCharges converted to float "
        f"({converted_nulls} non-numeric values set to NaN)"
    )

    return df


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


def encode_binary_columns(
    df: pd.DataFrame, columns: List[str] = binary_columns
) -> pd.DataFrame:
    """
    Encode binary variables using Label Encoding.

    Args:
        df: Input DataFrame with binary columns.
        columns: List of column names to encode.

    Returns:
        DataFrame with specified binary columns encoded as integers.

    Note:
    This modifies a copy of the DataFrame, never the original.
        For production, consider OrdinalEncoder or OneHotEncoder
        inside an sklearn Pipeline (covered on Day 10).
    """
    df_encoded = df.copy()
    encoder = LabelEncoder()

    for column in binary_columns:
        if column not in df_encoded.columns:
            logger.warning(f"Column '{column}' not found — skipping")
            continue

        df_encoded[column] = encoder.fit_transform(df_encoded[column])
        logger.info(f"Encoded column: '{column}'")

    return df_encoded


def encode_multiclass_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode columns with more than 2 categories.

    Columns like InternetService have values:
    'DSL', 'Fiber optic', 'No' — these need dummy encoding.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with multi-class columns one-hot encoded.
    """
    # These columns have 3+ unique values
    multiclass_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    cols_present = [c for c in multiclass_cols if c in df.columns]

    df_encoded = pd.get_dummies(
        df,
        columns=cols_present,
        drop_first=True,  # avoid multicollinearity
    )

    logger.info(
        f"One-hot encoded {len(cols_present)} columns. "
        f"New shape: {df_encoded.shape}"
    )
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


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Controls the shuffling applied to the data before applying the split.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train Size :{X_train.shape[0]} " f"Test Size :{X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Run all preprocessing steps in the correct order.

    This is the ONLY function train.py needs to call.
    It chains all steps and returns train/test splits.

    Args:
        df           : Raw loaded DataFrame.
        test_size    : Fraction for test set.
        random_state : Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Example:
        >>> df = load_csv("data/raw/churn.csv")
        >>> X_train, X_test, y_train, y_test = (
        ...     run_preprocessing_pipeline(df)
        ... )
    """
    logger.info("Starting preprocessing pipeline...")

    df = drop_useless_columns(df)
    df = fix_total_charges(df)
    df = handle_missing_values(df)
    df = encode_binary_columns(df)
    df = encode_multiclass_columns(df)

    X, y = split_features_and_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)

    logger.info("Preprocessing pipeline complete ✓")
    return X_train, X_test, y_train, y_test
