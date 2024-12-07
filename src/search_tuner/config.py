import os
from typing import Optional, Any

import configargparse

from src.utils.logger_setup import get_logger

# Create a logger for this module
logger = get_logger(__name__)


class Config:
    def __init__(self):
        self._parser = configargparse.ArgParser(
            description="search tuner: find an appropriate value for the faiss nprobes variable for a given set of queries, a ground truth and an IVF index.",
            default_config_files=["config_search_tuner.ini"],
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
        self._validate_file_path("index", [".index"])
        self._validate_file_path("queries", [".csv", ".tsv"])
        self._validate_directory_path("work_dir")
        self._validate_file_path("gt", [".csv", ".tsv"])
        self._validate_positive_int("min_nprobes", True)
        self._validate_positive_int("max_nprobes", True)
        self._validate_positive_int("step_size", True)
        self._log_parameters()

    def _define_arguments(self) -> None:
        """
        Define the command-line and config file arguments.
        """
        # IO arguments
        self._parser.add_argument(
            "--queries",
            required=True,
            help=(
                "File containing the queries (Supports .csv and .tsv)"
            ),
            type=str,
            action='store',
            dest="queries",
            metavar="<path>",
        )

        self._parser.add_argument(
            "--index",
            required=True,
            help=(
                "File containing a FAISS index (supports faiss index types 'IndexIDMap(IndexFlatIP)', IndexIVFFlat)"
            ),
            type=str,
            action='store',
            dest="index",
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
            "--gt",
            required=True,
            help=(
                "File containing the reference ranking used for evaluation. Supports .csv and .tsv"
            ),
            type=str,
            action='store',
            dest="gt",
            metavar="<path>",
        )

        self._parser.add_argument(
            "--min_nprobes",
            required=True,
            help=(
                "Minimum number of clusters to probe during FAISS nprobes tuning."
            ),
            type=int,
            action='store',
            dest="min_nprobes",
        )

        self._parser.add_argument(
            "--max_nprobes",
            required=True,
            help=(
                "Maximum number of clusters to probe during FAISS nprobes tuning."
            ),
            type=int,
            action='store',
            dest="max_nprobes",
        )

        self._parser.add_argument(
            "--step_size",
            required=True,
            help="Specifies the increment for the number of nprobes tested. "
                 "For example, a step size of 5 will test nprobes in increments of 5 (e.g., 5, 10, 15, ...).",
            type=int,
            action='store',
            dest="step_size",
            metavar="<int>",
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

    def _validate_file_path(self, param: str, required_extensions: Optional[list] = None) -> None:
        """
        Validate that a file path exists and optionally check if it has one of the required extensions.

        Parameters:
        ----------
        param : str
            The name of the parameter to validate.

        required_extensions : Optional[list], optional
            A list of valid extensions to check the file against. If provided, ensures that the file ends with one of them.
        """
        path = self.get(param)
        if not os.path.isfile(path):  # Check if the path is a valid file
            raise FileNotFoundError(f"--{param}: Path '{path}' is not a valid file.")

        if required_extensions:
            if not any(path.endswith(ext) for ext in required_extensions):
                valid_extensions = ', '.join(required_extensions)
                raise ValueError(
                    f"--{param}: Path '{path}' does not end with one of the following extensions: {valid_extensions}"
                )

    def _validate_positive_int(self, param: str, strict: bool = False) -> None:
        value = self.get(param)
        int_value = int(value)
        if (strict and int_value <= 0) or (not strict and int_value < 0):
            comparison = "greater than 0" if strict else "0 or greater"
            raise ValueError(f"--{param}: {value} is not a positive integer ({comparison}).")

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
