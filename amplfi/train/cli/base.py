from lightning.pytorch.cli import LightningCLI

from ..callbacks import SaveConfigCallback
import torch


class AmplfiBaseCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        # hack into init to hardcode
        # parser_mode to omegaconf for all subclasses
        kwargs["parser_kwargs"] = {"parser_mode": "omegaconf"}
        kwargs["save_config_callback"] = SaveConfigCallback
        kwargs["save_config_kwargs"] = {"overwrite": True}
        super().__init__(*args, **kwargs)

    def after_instantiate_classes(self) -> None:
        super().after_instantiate_classes()
        torch.set_float32_matmul_precision(
            self._get(self.config, "matmul_precision")
        )

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--matmul_precision", type=str, default="highest")
        parser.link_arguments(
            "data.init_args.sample_rate",
            "data.init_args.waveform_sampler.init_args.sample_rate",
            apply_on="parse",
        )

        parser.link_arguments(
            "data.init_args.kernel_length",
            "data.init_args.waveform_sampler.init_args.kernel_length",
            apply_on="parse",
        )

        parser.link_arguments(
            "data.init_args.fduration",
            "data.init_args.waveform_sampler.init_args.fduration",
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

        parser.link_arguments(
            "seed_everything",
            "data.init_args.seed",
            apply_on="parse",
        )
