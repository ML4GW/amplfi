import os

from amplfi.law.paths import paths


class AmplfiDataTaskMixin:
    """
    Mixin class for appending additional apptainer
    and/or condor environment variable to `mldatafind.law.DataTask`

    This class should be inherited
    alongside a subclass of `mldatafind.law.DataTask`
    """

    def __init__(self, *args, **kwargs):
        # calls `mldatafind.law.DataTask` constructor
        super().__init__(*args, **kwargs)
        self.image = paths().container_root / "data.sif"

    def sandbox_env(self, env: dict[str, str]) -> dict[str, str]:
        """
        Append amplfi specific environment variables to the sandbox environment
        """
        # get environment variables defined by
        # `mldatafind.law.DataTask` parent class
        env = super().sandbox_env(env)

        # append amplfi specific environment variables
        for envvar, value in os.environ.items():
            if envvar.startswith("AMPLFI_"):
                env[envvar] = value
        return env

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
