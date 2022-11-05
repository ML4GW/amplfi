import logging
import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="function")
def datadir():
    tmpdir = Path(__file__).resolve().parent / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    logging.shutdown()
    shutil.rmtree(tmpdir)


@pytest.fixture(scope="function")
def logdir():
    tmpdir = Path(__file__).resolve().parent / "log"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    logging.shutdown()
    shutil.rmtree(tmpdir)
