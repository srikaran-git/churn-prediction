# main.py
"""
Entry point for the Churn Prediction system.
Run this file to train the model and evaluate results.
"""

# Standard library imports
from pathlib import Path

# Local imports
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main pipeline execution function."""
    logger.info("=" * 50)
    logger.info("Churn Prediction System — Starting")
    logger.info("=" * 50)

    # Load configuration
    config = load_config()
    logger.info(f"Project: {config['project']['name']} v{config['project']['version']}")
    logger.info("Pipeline ready. More steps coming in Day 4!")
    logger.info(f"Author    : {config['project']['author']}")

    # Verify data path exists(we'll use real data on Day 4)
    raw_data_path = Path(config["data"]["raw_data_path"])
    if not raw_data_path.exists():
        logger.warning(f"Raw data path does not exist: {raw_data_path}")
        logger.warning("Add your CSV file there before running Day 4.")
    else:
        logger.info(f"Data found at: {raw_data_path}")

    logger.info("Day 3 complete — Clean code structure in place!")


if __name__ == "__main__":
    main()
