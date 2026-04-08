# src/data/data_loader.py
"""
Data loading module for the churn prediction pipeline.

Responsibility: ONLY loading data from disk into memory.
Nothing else. No cleaning, no transforming — just loading.
"""

# ── Standard library ──────────────────────────────────────────
from pathlib import Path
from typing import Optional

# ── Third-party ───────────────────────────────────────────────
import pandas as pd

# ── Internal ──────────────────────────────────────────────────
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_csv(
    file_path: str, encoding="utf-8", nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path : Path to the CSV file.
        encoding  : File encoding. Defaults to utf-8.
        nrows     : Load only first N rows (useful for testing).
                    Pass None to load everything.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError : If the file path doesn't exist.
        ValueError        : If the loaded file has no rows.

    Example:
        >>> df = load_csv("data/raw/churn.csv")
        >>> df = load_csv("data/raw/churn.csv", nrows=100)
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"File not found: {file_path}\n" f"File not found: {file_path}"
        )

    logger.info(f"Loading data from :{file_path}")
    df = pd.read_csv(file_path, encoding=encoding, nrows=nrows)
    if df.empty:
        raise ValueError(f"Loaded file is empty: {file_path}\n")

    logger.info(
        f"Loaded data from {file_path} with shape {df.shape} and columns: {df.columns.tolist()}"
    )

    return df


def get_data_summary(df: pd.DataFrame) -> None:
    """
    Log a quick summary of the DataFrame for debugging.

    This is NOT a transformation — it only logs information.
    Call this after loading to verify data looks right.

    Args:
        df: DataFrame to summarize."""
    summary = (
        f"Data Summary:\n"
        f"Shape: {df.shape}\n"
        f"Columns: {df.columns.tolist()}\n"
        f"Data Types:\n{df.dtypes}"
        f"Missing Values:\n{df.isnull().sum()}"
        f"Unique Values:\n{df.nunique()}"
        f"Duplicate Rows: {df.duplicated().sum()}"
        f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB\n"
    )
    logger.info(summary)
