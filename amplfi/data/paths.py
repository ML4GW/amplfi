import os

import luigi
from mldatafind.law.parameters import PathParameter


class paths(luigi.Config):
    data_dir = PathParameter(default=os.getenv("AMPLFI_DATADIR", ""))
    condor_dir = PathParameter(default=os.getenv("AMPLFI_CONDORDIR", ""))
    container_root = PathParameter(
        default=os.getenv("AMPLFI_CONTAINER_ROOT", "")
    )
