# src/utils/logger.py

import logging
import os
from typing import Optional

def setup_logger(
    name: str = "visdrone_logger",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Configure and return a standard Python Logger object that supports output to console and/or file.

    Args:
        name (str): The name of the logger, e.g., "visdrone_logger".
        log_file (Optional[str]): The path to the log file, e.g., "logs/visdrone_parse_log.txt". 
                                  If None, logs will only be output to the console.
        log_level (int): The logging level, default is logging.INFO.
        log_format (str): The log message format string.

    Returns:
        logging.Logger: A configured Logger instance.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers (prevent duplicate log messages)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(log_format)

    # --- Output to console ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- If log_file is provided, also output to file ---
    if log_file:
        # Ensure the log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:  # e.g., if log_file is "logs/xxx.log"
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger