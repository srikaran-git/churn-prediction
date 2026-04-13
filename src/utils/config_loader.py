# src/utils/config_loader.py

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load YAML config file and return as a dictionary.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary containing all config values.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded from: {config_path}")
    return config


def load_env(env_path: str = ".env") -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_path: Path to the .env file.
    """
    path = Path(env_path)

    if path.exists():
        load_dotenv(dotenv_path=path)
        logger.info(f"Environment variables loaded from: {env_path}")
    else:
        logger.warning(
            f".env file not found at {env_path}. " "Using system environment variables."
        )


def get_env(key: str, default: str = None) -> str:
    """
    Safely retrieve an environment variable.

    Args:
        key: The environment variable name.
        default: Default value if key is not found.

    Returns:
        The value of the environment variable.
    """
    value = os.getenv(key, default)

    if value is None:
        logger.warning(f"Environment variable '{key}' not set and no default provided.")

    return value
