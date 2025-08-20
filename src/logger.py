import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

# Generate log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create console handler for displaying logs in console as well
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Get logger and add console handler
logger = logging.getLogger(__name__)
logger.addHandler(console_handler)

def get_logger(name: str = __name__):
    """
    Get a configured logger instance

    Args:
        name (str): Name of the logger

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Prevent adding multiple handlers
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(LOG_FILE_PATH)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

    return logger

def log_dataframe_info(df, df_name: str, logger_instance=None):
    """
    Log basic information about a DataFrame

    Args:
        df: pandas DataFrame
        df_name (str): Name/description of the DataFrame
        logger_instance: Logger instance to use
    """
    if logger_instance is None:
        logger_instance = get_logger()

    logger_instance.info(f"DataFrame '{df_name}' Info:")
    logger_instance.info(f"Shape: {df.shape}")
    logger_instance.info(f"Columns: {list(df.columns)}")
    logger_instance.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    if hasattr(df, 'isnull'):
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger_instance.info(f"Null values: {null_counts[null_counts > 0].to_dict()}")
        else:
            logger_instance.info("No null values found")

def log_model_performance(metrics: dict, model_name: str, logger_instance=None):
    """
    Log model performance metrics

    Args:
        metrics (dict): Dictionary containing performance metrics
        model_name (str): Name of the model
        logger_instance: Logger instance to use
    """
    if logger_instance is None:
        logger_instance = get_logger()

    logger_instance.info(f"Model Performance - {model_name}:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger_instance.info(f"{metric}: {value:.4f}")
        else:
            logger_instance.info(f"{metric}: {value}")

# Create a default logger instance
default_logger = get_logger(__name__)

