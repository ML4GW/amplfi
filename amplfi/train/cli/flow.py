from ..data.datasets.base import AmplfiDataset
from ..models.base import AmplfiModel
from .base import AmplfiBaseCLI


class AmplfiFlowCLI(AmplfiBaseCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.link_arguments(
            "data.init_args.inference_params",
            "model.init_args.arch.init_args.num_params",
            compute_fn=lambda x: len(x),
            apply_on="parse",
        )

        parser.link_arguments(
            "data.init_args.ifos",
            "model.init_args.arch.init_args.embedding_net.init_args.num_ifos",
            compute_fn=lambda x: len(x),
            apply_on="parse",
        )

        return parser


def main(args=None):
    cli = AmplfiFlowCLI(
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
