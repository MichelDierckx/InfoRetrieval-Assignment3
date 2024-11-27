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
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for the given module name.
    Each logger is prefixed with the module's name for clarity.
    """
    return logging.getLogger(module_name)
