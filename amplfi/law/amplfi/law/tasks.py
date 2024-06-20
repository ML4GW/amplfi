import law

from amplfi.law.base import AmplfiDataTaskMixin
from amplfi.law.paths import paths
from mldatafind.law.tasks import Fetch
from mldatafind.law.tasks import Query as _Query


# add mixin for appending amplfi specific
# environment variables to condor and containers
class Query(AmplfiDataTaskMixin, _Query):
    pass


# override FetchTrain and FetchTest
# tasks to use Query with mixin
class FetchTrain(AmplfiDataTaskMixin, Fetch):
    def workflow_requires(self):
        reqs = {}
        reqs["segments"] = Query.req(self, segments_file=self.segments_file)
        return reqs


class FetchTest(AmplfiDataTaskMixin, Fetch):
    def workflow_requires(self):
        reqs = {}
        reqs["segments"] = Query.req(self, segments_file=self.segments_file)
        return reqs


class DataGeneration(law.WrapperTask):
    """
    Pipeline for launching FetchTrain and FetchTest tasks
    """

    def requires(self):
        yield FetchTrain.req(
            self,
            data_dir=paths().data_dir / "train" / "background",
            condor_directory=paths().condor_dir / "train",
        )

        yield FetchTest.req(
            self,
            data_dir=paths().data_dir / "test" / "background",
            condor_directory=paths().condor_dir / "test",
        )
