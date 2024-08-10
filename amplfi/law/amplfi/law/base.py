import os
from pathlib import Path

import law

from amplfi.law.paths import paths
from mldatafind.law.base import DataSandbox

root = Path(__file__).resolve().parent.parent.parent.parent.parent
DATA_SANDBOX = f"amplfi::{paths().container_root / 'data.sif'}"


class AmplfiDataSandbox(DataSandbox):
    """
    Subclass of `mldatafind.law.base.DataSandbox` for running amplfi tasks;

    Appends amplfi specific environment variables to the container
    and mounts the local amplfi repo into the container in dev mode
    """

    sandbox_type = "amplfi"

    def _get_volumes(self):
        # if running in dev mode, mount the local
        # mldatafind repo into the container so
        # python code changes are reflected
        volumes = super()._get_volumes()
        if self.task and getattr(self.task, "dev", False):
            volumes[str(root)] = "/opt/amplfi"
        return volumes

    def _get_env(self):
        env = super()._get_env()

        # append amplfi specific environment variables
        for envvar, value in os.environ.items():
            if envvar.startswith("AMPLFI_"):
                env[envvar] = value
        return env


law.config.update(AmplfiDataSandbox.config())


class AmplfiDataTaskMixin:
    """
    Mixin class for appending additional apptainer
    and/or condor environment variable to `mldatafind.law.DataTask`

    This class should be inherited
    alongside a subclass of `mldatafind.law.DataTask`
    """

    def build_environment(self) -> str:
        """
        Append amplfi specific environment variables to the condor submit file
        """
        # get `mldatafind.law.condor.LDGCondorWorkflow` environment variables
        environment = super().build_environment()

        # forward any env variables that start with AMPLFI_
        # that the law config may need to parse
        for envvar, value in os.environ.items():
            if envvar.startswith("AMPLFI_"):
                environment += f"{envvar}={value} "
        environment += '"'
        return environment
