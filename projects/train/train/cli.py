from lightning.pytorch.cli import LightningCLI
from train.data.datasets.base import AmplfiDataset
from train.models.base import AmplfiModel


class AmplifiCLI(LightningCLI):
    def link_flow_arguments(self, parser):
        parser.link_arguments(
            "data.init_args.inference_params",
            "model.init_args.arch.init_args.num_params",
            compute_fn=lambda x: len(x),
            apply_on="parse",
        )

        parser.link_arguments(
            "model.init_args.arch.init_args.context_dim",
            "model.init_args.arch.init_args.embedding_net.init_args.context_dim",  # noqa
            apply_on="parse",
        )

        parser.link_arguments(
            "data.init_args.ifos",
            "model.init_args.arch.init_args.embedding_net.init_args.num_ifos",
            compute_fn=lambda x: len(x),
            apply_on="parse",
        )

        return parser

    def link_waveform_sampler_arguments(self, parser):
        parser.link_arguments(
            "data.init_args.sample_rate",
            "data.init_args.waveform_sampler.init_args.sample_rate",
            apply_on="parse",
        )

        parser.link_arguments(
            ("data.init_args.kernel_length", "data.init_args.fduration"),
            "data.init_args.waveform_sampler.init_args.duration",
            compute_fn=lambda *x: sum(x),
            apply_on="parse",
        )

        parser.link_arguments(
            "data.init_args.inference_params",
            "data.init_args.waveform_sampler.init_args.inference_params",
            apply_on="parse",
        )
        return parser

    def add_arguments_to_parser(self, parser):
        parser = self.link_waveform_sampler_arguments(parser)
        parser = self.link_flow_arguments(parser)

        parser.link_arguments(
            "data.init_args.inference_params",
            "model.init_args.inference_params",
            apply_on="parse",
        )


def main(args=None):
    # any subclasses of AmplifiModel and BaseDataset
    # will automatically be registered with the CLI
    # and their arguments will be available at
    # the command line
    cli = AmplifiCLI(
        AmplfiModel,
        AmplfiDataset,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=101588,
        args=args,
    )
    return cli


if __name__ == "__main__":
    main()
