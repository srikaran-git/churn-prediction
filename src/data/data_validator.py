# src/data/data_validator.py
"""
Data validation module.
Checks that incoming data meets minimum quality requirements
before it enters the preprocessing pipeline.
"""

from typing import List

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Define minimum requirements
REQUIRED_COLUMNS = [
    "gender",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]
MIN_ROWS_REQUIRED = 100


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str] = REQUIRED_COLUMNS,
) -> bool:
    """
    Check that all required columns exist in the DataFrame.

    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must be present.

    Returns:
        True if all columns exist, False otherwise.
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    logger.info("All required columns present.")
    return True


def validate_minimum_rows(
    df: pd.DataFrame,
    min_rows: int = MIN_ROWS_REQUIRED,
) -> bool:
    """
    Ensure DataFrame has enough rows for meaningful training.

    Args:
        df: DataFrame to check.
        min_rows: Minimum acceptable number of rows.

    Returns:
        True if row count is sufficient, False otherwise.
    """
    row_count = df.shape[0]
    if row_count < min_rows:
        logger.error(f"Not enough rows: {row_count} found, but {min_rows} required.")
        return False
    logger.info(f"Row count sufficient: {row_count} rows.")
    return True


def validate_no_duplicate_rows(df: pd.DataFrame) -> bool:
    """
    Warn if DataFrame contains duplicate rows.

    Args:
        df: DataFrame to check.

    Returns:
        True if no duplicates, False if duplicates found.
    """
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        logger.warning(
            f"Duplicate rows found in data: {duplicate_count}"
            f" duplicate rows ({duplicate_count/len(df):.1%} total)"
        )
        return False
    logger.info("No duplicate rows found.")
    return True


def run_all_validations(df: pd.DataFrame) -> bool:
    """
    Run all validation checks and return overall pass/fail.

    Args:
        df: DataFrame to validate.

    Returns:
        True only if ALL validations pass.
    """
    checks = [
        validate_required_columns(df),
        validate_minimum_rows(df),
        validate_no_duplicate_rows(df),
    ]
    if all(checks):
        logger.info("All data validation checks passed.")
        return True
    logger.error("Data validation failed. See logs for details.")
    return False
