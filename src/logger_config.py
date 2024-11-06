"""
Logger Configuration Module

This module provides a centralized configuration for logging across the entire project.
It sets up console and file handlers with appropriate formatting and log levels.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(base_log_name: Optional[str] = None) -> None:
    """
    Configure the logging system for the entire project.

    This function sets up both console and file handlers with appropriate
    formatting and log levels. It should be called once at the start of the application.

    Args:
        log_file (str, optional): Name of the log file. Defaults to "app_<timestamp>.log".
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = base_log_name or 'app'
    log_file = f"{base_name}_{timestamp}.log"
    log_file_path = log_dir / log_file

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Prevent logging messages from being propagated to the root logger
    root_logger.propagate = False

    logging.info(f"Logging system initialized. Log file: {log_file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name (str): Name of the logger, typically __name__ of the module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)
