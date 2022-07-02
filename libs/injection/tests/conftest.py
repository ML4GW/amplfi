import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="function")
def tmpdir():
    tmpdir = Path(__file__).resolve().parent / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=False)
    yield tmpdir
    shutil.rmtree(tmpdir)
