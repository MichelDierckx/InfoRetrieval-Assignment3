import logging

LOG_FILE = "logs/app.log"


def setup_logging():
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Set up a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to the handler
    console_handler.setFormatter(formatter)

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.addHandler(console_handler)


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for the given module name.
    Each logger is prefixed with the module's name for clarity.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    return logger


def configure_file_logger(logger: logging.Logger, log_filepath: str) -> None:
    """
    Configures the given logger to write logs to the specified file.

    :param logger: The logger instance to configure.
    :param log_filepath: The path to the logfile.
    """
    # Create a file handler
    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)
