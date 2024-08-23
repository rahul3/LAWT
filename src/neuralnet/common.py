import logging
import math

def get_logger(name=None, level=logging.INFO, log_file=None):
    """
    Creates and returns a logger with the specified name, level, and optional file handler.
    
    Args:
        name (str): Name of the logger. If None, the root logger is returned.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file (str): Optional. If provided, logs will be written to this file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if the logger already has handlers (to avoid adding duplicate handlers)
    if not logger.hasHandlers():
        # Create console handler and set level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create a formatter and set it for the console handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add the console handler to the logger
        logger.addHandler(console_handler)
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def log_loss(loss):
    return -math.log10(loss + 1e-8) 
