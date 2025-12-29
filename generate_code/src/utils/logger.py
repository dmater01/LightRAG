import logging
import os
import sys
from datetime import datetime

def setup_logger(name="benchmark", log_dir="logs", level=logging.INFO):
    """
    Sets up a logger that writes to both console and a file.
    
    Args:
        name (str): Name of the logger.
        log_dir (str): Directory to save log files.
        level (int): Logging level (e.g., logging.INFO).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized. Log file: {log_file}")

    return logger
