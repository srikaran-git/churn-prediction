# src/utils/env_validator.py

import os

from src.utils.exceptions import ConfigurationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_ENV_VARS = [
    "ENV",
    "LOG_LEVEL",
    "MODEL_PATH",
]

VALID_ENVIRONMENTS = {"development", "production"}


def validate_environment() -> None:
    """
    Validate all required environment variables are present and valid.
    Collects ALL missing vars before raising — no guessing which one failed.
    Raises ConfigurationError on the first problem found.
    """
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]

    if missing:
        logger.error("Missing required environment variables: %s", missing)
        raise ConfigurationError(
            f"Missing required environment variables: {missing}. "
            f"Check your .env file against .env.example."
        )

    env = os.getenv("ENV")
    if env not in VALID_ENVIRONMENTS:
        logger.error(
            "Invalid ENV value '%s'. Must be one of: %s", env, VALID_ENVIRONMENTS
        )
        raise ConfigurationError(
            f"Invalid ENV='{env}'. Must be one of: {VALID_ENVIRONMENTS}"
        )

    logger.info(
        "Environment validated | ENV=%s | LOG_LEVEL=%s | MODEL_PATH=%s",
        os.getenv("ENV"),
        os.getenv("LOG_LEVEL"),
        os.getenv("MODEL_PATH"),
    )
