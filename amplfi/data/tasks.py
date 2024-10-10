import law
import luigi
from mldatafind.law.tasks import Fetch
from mldatafind.law.tasks import Query as _Query

from .base import DATA_SANDBOX, AmplfiDataTaskMixin
from .paths import paths


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

    dev = luigi.BoolParameter(
        default=False, description="Run the task in development mode."
    )

    def requires(self):
        yield FetchTrain.req(
            self,
            sandbox=DATA_SANDBOX,
            data_dir=paths().data_dir / "train" / "background",
            condor_directory=paths().condor_dir / "train",
        )

        yield FetchTest.req(
            self,
            sandbox=DATA_SANDBOX,
            data_dir=paths().data_dir / "test" / "background",
            condor_directory=paths().condor_dir / "test",
        )
