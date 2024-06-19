from mldatafind.law.parameters import PathParameter
import os
import luigi

class paths(luigi.Config):
    data_dir = PathParameter(default=os.getenv("AMPLFI_DATADIR"))
    condor_dir = PathParameter(default=os.getenv("AMPLFI_CONDOR_DIR"))