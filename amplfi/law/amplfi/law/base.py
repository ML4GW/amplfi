import os

from amplfi.law.paths import paths


class AmplfiDataTaskMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = paths().container_root / "data.sif"

    def sandbox_env(self, env):
        # get `mldatafind.law.DataTask` environment variables
        env = super().sandbox_env(env)

        # append amplfi specific environment variables
        for envvar, value in os.environ.items():
            if envvar.startswith("AMPLFI_"):
                env[envvar] = value
        return env

    def build_environment(self) -> str:
        # get `mldatafind.law.condor.LDGCondorWorkflow` environment
        environment = super().build_environment()

        # forward any env variables that start with AMPLFI_
        # that the law config may need to parse
        for envvar, value in os.environ.items():
            if envvar.startswith("AMPLFI_"):
                environment += f"{envvar}={value} "
        environment += '"'
        return environment
