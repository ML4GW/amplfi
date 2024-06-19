import law
from mldatafind.law.tasks import Fetch

from amplfi.law.paths import paths


# rename for clarity in config file
class FetchTrain(Fetch):
    pass


class FetchTest(Fetch):
    pass


class DataGeneration(law.WrapperTask):
    def requires(self):
        yield FetchTrain.req(
            self,
            data_dir=paths().data_dir / "train" / "background",
            condor_dir=paths().condor_dir / "train",
        )
        yield FetchTest.req(
            self,
            data_dir=paths().data_dir / "test" / "background",
            condor_dir=paths().condor_dir / "test",
        )
