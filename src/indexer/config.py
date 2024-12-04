import os
from typing import Optional, Any

import configargparse

from src.utils.logger_setup import get_logger

# Create a logger for this module
logger = get_logger(__name__)


class Config:
    def __init__(self):
        self._parser = configargparse.ArgParser(
            description="Indexer: Create an index for given embeddings using FAISS.",
            default_config_files=["config.ini"],
            args_for_setting_config_path=["-c", "--config"],
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )
        self._define_arguments()
        self._namespace = None

    def get(self, option: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Retrieve configuration options with a default value if the option does not exist.
        :param option: str, the configuration option to retrieve.
        :param default: Optional[str], the default value to return if the option does not exist. Defaults to None.
        :return: Optional[str], The value of the configuration option or the default value.
        """
        if self._namespace is None:
            raise RuntimeError("The configuration has not been initialized. Call `parse()` first.")
        return self._namespace.get(option, default)

    def parse(self, args_str: Optional[str] = None) -> None:
        """
        Parse the configuration settings.

        Parameters
        ----------
        args_str : Optional[str]
            If None, arguments are taken from sys.argv; otherwise, a string of arguments.
            Arguments not specified on the command line are taken from the config file.
        """
        if self._namespace is not None:
            return  # Skip parsing if already parsed

        # Parse the arguments
        self._namespace = vars(self._parser.parse_args(args_str))
        self._validate_directory_path("embeddings")
        self._validate_directory_path("work_dir")
        self._log_parameters()

    def _define_arguments(self) -> None:
        """
        Define the command-line and config file arguments.
        """
        # IO arguments
        self._parser.add_argument(
            "--embeddings",
            required=True,
            help="Directory containing the embeddings (lance dataset) to be indexed.",
            type=str,
            action='store',
            dest="embeddings",
            metavar="<path>",
        )

        self._parser.add_argument(
            "--work_dir",
            required=True,
            help="Working directory, used to write temporary files and output files.",
            type=str,
            action='store',
            dest="work_dir",
            metavar="<path>",
        )

        self._parser.add_argument(
            "--index_filename",
            required=True,
            help="Filename of the FAISS index to be created.",
            type=str,
            action='store',
            dest="index_filename",
        )

    def _validate_directory_path(self, param: str, required_extensions: Optional[list] = None) -> None:
        """
        Validate that a directory path exists and optionally check if it has one of the required extensions.

        Parameters:
        ----------
        param : str
            The name of the parameter to validate.

        required_extensions : Optional[list], optional
            A list of file extensions to validate against. If provided, checks that at least one file
            in the directory matches one of the extensions.
        """
        path = self.get(param)
        if not os.path.isdir(path):  # Check if path is a valid directory
            raise NotADirectoryError(f"--{param}: Path '{path}' is not a valid directory.")

        if required_extensions:
            # Check if any file in the directory matches the required extensions
            files_with_valid_extension = [
                file for file in os.listdir(path) if any(file.endswith(ext) for ext in required_extensions)
            ]
            if not files_with_valid_extension:
                valid_extensions = ', '.join(required_extensions)
                raise ValueError(
                    f"--{param}: No files with extensions {valid_extensions} found in the directory '{path}'.")

    def _log_parameters(self) -> None:
        """Log all chosen parameters."""
        for key, value in self._namespace.items():
            logger.debug(f"{key}: {value}")

    def __getattr__(self, option):
        """
        Retrieve configuration options as attributes.

        Raises a KeyError with a helpful message if the option does not exist.
        """
        if self._namespace is None:
            raise RuntimeError("The configuration has not been initialized. Call `parse()` first.")
        if option not in self._namespace:
            raise KeyError(f"The configuration option '{option}' does not exist.")
        return self._namespace[option]

    def __getitem__(self, item):
        return self.__getattr__(item)
