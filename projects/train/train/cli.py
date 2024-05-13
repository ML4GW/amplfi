from lightning.pytorch.cli import LightningCLI
from train.data.datasets.base import AmplfiDataset
from train.models.base import AmplfiModel


class AmplfiBaseCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
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

        parser.link_arguments(
            "data.init_args.inference_params",
            "model.init_args.inference_params",
            apply_on="parse",
        )


class AmplfiFlowCli(AmplfiBaseCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
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


class AmplfiSimilarityCli(AmplfiBaseCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)

        parser.link_arguments(
            "data.init_args.ifos",
            "model.init_args.arch.init_args.num_ifos",
            compute_fn=lambda x: len(x),
            apply_on="parse",
        )
        return parser


def main(args=None):

    cli = AmplfiFlowCli(
        AmplfiModel,
        AmplfiDataset,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        seed_everything_default=101588,
        args=args,
        parser_kwargs={"default_env": True},
    )
    return cli


if __name__ == "__main__":
    main()
