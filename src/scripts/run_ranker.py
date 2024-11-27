from typing import Union, List

import lance

from src.ranker.config import Config
from src.utils.logger_setup import setup_logging


def main(args: Union[str, List[str]] = None) -> int:
    ds = lance.dataset("fake")
    setup_logging()
    config = Config()
    config.parse(args)
    return 0


if __name__ == "__main__":
    main()
