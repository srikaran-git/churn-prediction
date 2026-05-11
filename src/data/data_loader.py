# src/data/data_loader.py
"""
Data loading module for the churn prediction pipeline.
Responsibility: ONLY loading data from disk into memory.
Nothing else. No cleaning, no transforming — just loading.
"""

# -- Standard library --
from pathlib import Path
from typing import Optional

# -- Third-party --
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessor import fix_total_charges

# -- Internal --
from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_csv(
    file_path: str,
    encoding: str = "utf-8",
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path : Path to the CSV file.
        encoding  : File encoding. Defaults to utf-8.
        nrows     : Load only first N rows. None = load all.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError   : If the file path does not exist.
        DataValidationError : If the file loads but is empty.
    """
    path = Path(file_path)

    if not path.is_file():
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info("Loading data from: %s", file_path)

    try:
        df = pd.read_csv(file_path, encoding=encoding, nrows=nrows)
    except Exception as e:
        logger.error("Unexpected error loading file: %s", file_path, exc_info=True)
        raise DataValidationError("Failed to read CSV file.") from e

    if df.empty:
        raise DataValidationError(f"Loaded file is empty: {file_path}")

    logger.debug("Loaded shape=%s | columns=%s", df.shape, df.columns.tolist())
    return df


def load_data(config: dict):
    """
    High-level loader used by the training pipeline.

    Reads the raw CSV using config paths, encodes the target,
    splits into train/test sets, and returns all four splits.

    Args:
        config: Loaded config dict from config.yaml.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    raw_path = config["data"]["raw_path"]
    target = config["data"]["target_column"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    df = load_csv(raw_path)
    df = fix_total_charges(df)

    if target not in df.columns:
        raise DataValidationError(
            f"Target column '{target}' not found in data. "
            f"Available columns: {df.columns.tolist()}"
        )

    X = df.drop(columns=[target])
    y = df[target]

    # --- Encode target BEFORE splitting to ensure y_train/y_test are numeric ---
    if set(y.dropna().unique()) == {"Yes", "No"}:
        y = y.map({"Yes": 1, "No": 0})
        logger.info("Target column encoded: Yes=1, No=0")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        "Split complete | Train: %s | Test: %s",
        X_train.shape,
        X_test.shape,
    )

    return X_train, X_test, y_train, y_test


def get_data_summary(df: pd.DataFrame) -> None:
    """
    Log a quick summary of the DataFrame for debugging.
    Call this after loading to verify data looks right.

    Args:
        df: DataFrame to summarize.
    """
    summary = (
        f"Data Summary:\n"
        f"  Shape        : {df.shape}\n"
        f"  Columns      : {df.columns.tolist()}\n"
        f"  Missing vals : {df.isnull().sum().to_dict()}\n"
        f"  Duplicates   : {df.duplicated().sum()}\n"
        f"  Memory       : {df.memory_usage(deep=True).sum() / 1024:.1f} KB"
    )
    logger.info(summary)
