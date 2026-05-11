import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_directory_exists(path: str) -> None:
    """
    Create directory if it doesn't exist.
    Should:
    - Accept a string path
    - Create the directory (including parent dirs)
    - Log a message if directory was created
    - NOT raise an error if directory already exists
    """
    if not path:
        raise ValueError("Path cannot be empty")

    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory created: {path}")
