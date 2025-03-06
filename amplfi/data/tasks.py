import law
import luigi
from mldatafind.law.parameters import PathParameter
from mldatafind.law.tasks import Fetch
from mldatafind.law.tasks import Query as _Query
from mldatafind.law.tasks.condor.workflows import StaticMemoryWorkflow
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


class LigoSkymap(
    AmplfiDataTaskMixin,
    law.LocalWorkflow,
    StaticMemoryWorkflow,
    law.SandboxTask,
):
    """
    Workflow for parallelizing skymap generation via ligo-skymap-from-samples
    """

    data_dir = PathParameter(
        description="Path to the directory containing the "
        "event sub directories."
        "Each sub directory should contain "
        "a posterior_samples.dat file."
    )
    ligo_skymap_args = luigi.ListParameter(
        description="Additional command line style arguments"
        "to pass to ligo-skymap-from-samples."
    )
    dev = luigi.BoolParameter(
        default=False, description="Run the task in development mode."
    )
    sandbox = DATA_SANDBOX

    def sandbox_env(self, env):
        env = super().sandbox_env(env)
        env.update({"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
        return env

    def create_branch_map(self):
        branch_map = {}
        for i, event_dir in enumerate(self.data_dir.iterdir()):
            branch_map[i] = event_dir / "posterior_samples.dat"
        return branch_map

    def output(self):
        event_dir = self.branch_data.parent
        return law.LocalFileTarget(event_dir / "skymap.fits")

    def run(self):
        from ligo.skymap.tool import ligo_skymap_from_samples

        args = [
            str(self.branch_data),
            "-j",
            str(self.request_cpus),
            "--maxpts",
            "10000",
            "-o",
            str(self.branch_data.parent),
        ]
        args.extend(self.ligo_skymap_args)
        ligo_skymap_from_samples.main(args)
