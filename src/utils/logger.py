# src/utils/logger.py

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str = "logs/app.log") -> logging.Logger:
    """
    Create and return a logger that writes to both
    the terminal and a log file simultaneously.

    Args:
        name      : Logger name — always pass __name__
        log_file  : Path to the log file

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Formatter ───────────────────────────────────────────────
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    # ── Handler 1: Terminal (stdout) ────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── Handler 2: File ─────────────────────────────────────────
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # File captures EVERYTHING
    file_handler.setFormatter(formatter)

    # ── Attach both handlers ─────────────────────────────────────
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
