import logging
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def datadir():
    tmpdir = Path(tempfile.gettempdir()).resolve() / "data"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    logging.shutdown()
    shutil.rmtree(tmpdir)


@pytest.fixture
def logdir():
    tmpdir = Path(tempfile.gettempdir()).resolve() / "log"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    logging.shutdown()
    shutil.rmtree(tmpdir)
