import law
import luigi
from mldatafind.law.parameters import PathParameter
from mldatafind.law.tasks import Fetch
from mldatafind.law.tasks import Query as _Query
from mldatafind.law.tasks.condor.workflows import StaticMemoryWorkflow
from .base import DATA_SANDBOX, AmplfiDataTaskMixin
from .paths import paths
from luigi.util import inherits


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
    ligo_skymap_args = luigi.OptionalListParameter(
        description="Additional command line style arguments"
        "to pass to ligo-skymap-from-samples.",
        default="",
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
        branch_map, i = {}, 0
        for event_dir in self.data_dir.iterdir():
            if event_dir.is_dir():
                branch_map[i] = event_dir / "posterior_samples.dat"
                i += 1
        return branch_map

    def output(self):
        event_dir = self.branch_data.parent
        return law.LocalFileTarget(event_dir / "skymap.fits")

    def run(self):
        from ligo.skymap.tool import (
            ligo_skymap_from_samples,
        )

        args = [
            str(self.branch_data),
            "-j",
            str(self.request_cpus),
            "-o",
            str(self.branch_data.parent),
        ]
        if self.ligo_skymap_args:
            args.extend(self.ligo_skymap_args)

        # call ligo-skymap-from-samples
        ligo_skymap_from_samples.main(args)


@inherits(LigoSkymap)
class AggregateLigoSkymap(
    AmplfiDataTaskMixin,
    law.SandboxTask,
):
    parameter_file = luigi.OptionalParameter(
        default="",
        description="Path to an hdf5 file containing `ra`, `dec` and `dist`"
        " datasets corresponding to the ground truth values of the event",
    )

    def requires(self):
        return LigoSkymap.req(self)

    def output(self):
        return law.LocalFileTarget(self.data_dir / "ligo_skymap_stats.hdf5")

    def run(self):
        from ligo.skymap.postprocess import crossmatch
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        import h5py

        crossmatch_attributes = [
            "searched_area",
            "searched_vol",
            "searched_prob",
            "searched_prob_vol",
            "searched_prob_dist",
            "offset",
            "contour_areas",
        ]
        data = {attr: [] for attr in crossmatch_attributes}
        for i, skymap in self.input()["collection"].items():
            with h5py.File(self.parameter_file, "r") as f:
                ra = f["ra"][i]
                dec = f["dec"][i]
                dist = f["dist"][i]

            coord = SkyCoord(
                ra=ra * u.rad,
                dec=dec * u.rad,
                distance=dist * u.Mpc,
                unit="rad",
            )
            cm = crossmatch(skymap, coord)
            for attr in crossmatch_attributes:
                data[attr].append(getattr(cm, attr))

        with h5py.File(self.output().path, "w") as f:
            for attr, values in data.items():
                f.create_dataset(attr, data=values)
