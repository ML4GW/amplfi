import law
import luigi
from mldatafind.law.parameters import PathParameter
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


class LigoSkymap(AmplfiDataTaskMixin):
    data_dir = PathParameter()
    sandbox = DATA_SANDBOX

    def create_branch_map(self):
        branch_map, i = {}, 1
        for i, f in enumerate(self.data_dir.iterdir()):
            branch_map[i] = f / "samples.dat"

        return branch_map

    def output(self):
        directory = self.branch_data.parent
        return law.LocalFileTarget(directory / "skymap.fits")

    def run(self):
        from ligo.skymap.tool import ligo_skymap_from_samples

        ligo_skymap_from_samples.main(
            [
                self.branch_data,
                "-j",
                self.request_cpus,
                "--maxpts",
                "10000",
                "-o",
                self.branch_data.parent,
            ]
        )
