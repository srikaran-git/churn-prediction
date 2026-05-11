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
]

target_column = "Churn"


def drop_useless_columns(
    df: pd.DataFrame, columns: List[str] = columns_to_drop
) -> pd.DataFrame:
    """
    Remove irrelevant columns from the DataFrame.

    Args:
        df      : Input DataFrame.
        columns : List of column names to drop.

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

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with TotalCharges as float.
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


def encode_binary_columns(
    df: pd.DataFrame, columns: List[str] = binary_columns
) -> pd.DataFrame:
    """
    Encode binary variables using Label Encoding.

    Args:
        df      : Input DataFrame with binary columns.
        columns : List of column names to encode.

    Returns:
        DataFrame with specified binary columns encoded as integers.
    """
    df_encoded = df.copy()
    encoder = LabelEncoder()

    for column in columns:
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


def preprocess(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply preprocessing to already-split train and test sets.

    This is what train.py calls. The split has already happened
    in load_data() — this function only transforms features.

    Critical rule: fit logic (like median fill value) is learned
    from X_train only, then applied to X_test. Never the reverse.

    Args:
        X_train : Raw training features (no target column).
        X_test  : Raw test features (no target column).

    Returns:
        Tuple of (X_train_processed, X_test_processed).
    """
    logger.info("Starting preprocessing...")

    X_train = X_train.copy()
    X_test = X_test.copy()

    # --- Drop useless columns ---
    X_train = drop_useless_columns(X_train)
    X_test = drop_useless_columns(X_test)

    # --- Fix TotalCharges ---
    X_train = fix_total_charges(X_train)
    X_test = fix_total_charges(X_test)

    # --- Fill missing values with train median (no data leakage) ---
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)  # <-- use train median, not test
    logger.info("Missing values filled using train set medians.")

    # --- Encode binary columns ---
    X_train = encode_binary_columns(X_train)
    X_test = encode_binary_columns(X_test)

    # --- One-hot encode multiclass columns ---
    X_train = encode_multiclass_columns(X_train)
    X_test = encode_multiclass_columns(X_test)

    # --- Align columns (test may be missing some dummies) ---
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    logger.info(
        "Preprocessing complete | Train: %s | Test: %s",
        X_train.shape,
        X_test.shape,
    )

    return X_train, X_test
