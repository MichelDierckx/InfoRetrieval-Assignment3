from typing import Union, List

from src.evaluator.config import Config
from src.evaluator.evaluator import run
from src.utils.logger_setup import setup_logging


def main(args: Union[str, List[str]] = None) -> int:
    setup_logging()
    config = Config()
    config.parse(args)
    run(config)
    return 0


if __name__ == "__main__":
    main()
