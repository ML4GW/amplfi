from lightning.pytorch.cli import LightningCLI


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
