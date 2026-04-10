import logging
from logging.handlers import RotatingFileHandler
import os
from configs.settings import config


def setup_logger(name: str):
    log_config = config["logging"]

    log_file = log_config["file"]
    max_bytes = log_config["max_bytes"]
    backup_count = log_config["backup_count"]
    level_str = log_config["level"]

    # Convert string level → logging constant
    level = getattr(logging, level_str.upper(), logging.INFO)

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)

    # Console handler (VERY useful for dev)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger