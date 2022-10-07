import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="function")
def datadir():
    tmpdir = Path(__file__).resolve().parent / "data"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture(scope="function")
def logdir():
    tmpdir = Path(__file__).resolve().parent / "log"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    shutil.rmtree(tmpdir)
