import logging
import sys
from typing import Optional


def configure_logging(
    filename: Optional[str] = None, verbose: bool = False
) -> None:
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )

    logger = logging.getLogger()
    if filename is not None:
        formatter = logging.Formatter(log_format)
        handler = logging.FileHandler(filename=filename, mode="w")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
