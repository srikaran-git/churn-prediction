from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Read the config.yaml file and return it as a dictionary.

    Example:
        config = load_config()
        print(config["project"]["name"])   # prints: churn-prediction
    """
    path = Path(config_path)

    # Check if the file actually exists before trying to open it
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at: {config_path}\n"
            "Make sure you are running commands from the project root folder."
        )

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    logger.info(f"Config loaded successfully from {config_path}")
    return config
