import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger .
    Use this in every file instead of print().

    How to use in any other file :
    from src.utils.logger import get_logger
    logger.info("Your message here")
    """
    logger = logging.getLogger(name)
    # Only add handlers if none exist (prevents duplicate log lines)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # This controls how each log line looks
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Handler 1: Show logs in your terminal
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        # Handler 2: Also save logs to a file
        Path("logs").mkdir(exist_ok=True)
        file_handler = logging.FileHandler("logs/app.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
