# main.py
"""
Entry point for the Churn Prediction system.
Run this file to train the model and evaluate results.
"""

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)


def main():
    """Main pipeline execution function."""
    logger.info("=" * 50)
    logger.info("Churn Prediction System — Starting")
    logger.info("=" * 50)

    # Load configuration
    config = load_config()
    logger.info(f"Project: {config['project']['name']} v{config['project']['version']}")
    logger.info("Pipeline ready. More steps coming in Day 4!")


if __name__ == "__main__":
    main()