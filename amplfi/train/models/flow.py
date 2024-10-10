import bilby
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..architectures.flows import FlowArchitecture
from ..callbacks import SaveAugmentedBatch
from ..testing import Result
from .base import AmplfiModel

Tensor = torch.Tensor


class FlowModel(AmplfiModel):
    """
    A LightningModule for training normalizing flows
    Args:
        arch:
            Neural network architecture to train.
            This should be a subclass of `FlowArchitecture`.
        learning_rate;
            Learning rate for the optimizer
        weight_decay:
            Weight decay for the optimizer
        save_top_k_models:
            Maximum number of best-performing model checkpoints
            to keep during training
        samples_per_event:
            Number of samples to draw per event for testing
    """

    def __init__(
        self,
        *args,
        arch: FlowArchitecture,
        samples_per_event: int = 20000,
        num_corner: int = 10,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        # construct our model
        self.model = arch
        self.samples_per_event = samples_per_event
        self.num_corner = num_corner

        # save our hyperparameters
        self.save_hyperparameters(ignore=["arch"])

        # if checkpoint is not None, load in model weights;
        # checkpint should only be specified here if running trainer.test
        self.maybe_load_checkpoint(self.checkpoint)

    def forward(self, strain, parameters) -> Tensor:
        return -self.model.log_prob(parameters, context=strain)

    def training_step(self, batch, _):
        strain, parameters = batch
        loss = self(strain, parameters).mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, _):
        strain, parameters = batch
        loss = self(strain, parameters).mean()
        self.log(
            "valid_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        return loss

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        callbacks.append(SaveAugmentedBatch())
        return callbacks

    def cast_as_bilby_result(
        self,
        samples: np.ndarray,
        truth: np.ndarray,
    ):
        """Cast posterior samples as Bilby Result object
        for ease of producing corner and pp plots

        Args:
            samples: posterior samples (1, num_samples, num_params)
            truth: true values of the parameters  (1, num_params)
            priors: dictionary of prior objects
            label: label for the bilby result object

        """

        injection_parameters = {
            k: float(v) for k, v in zip(self.inference_params, truth)
        }

        # create dummy prior with correct attributes
        # for making our results compatible with bilbys make_pp_plot
        priors = {
            param: bilby.core.prior.base.Prior(latex_label=param)
            for param in self.inference_params
        }
        posterior = dict()
        for idx, k in enumerate(self.inference_params):
            posterior[k] = samples.T[idx].flatten()
        posterior = pd.DataFrame(posterior)

        r = Result(
            label="PEModel",
            injection_parameters=injection_parameters,
            posterior=posterior,
            search_parameter_keys=self.inference_params,
            priors=priors,
        )
        return r

    def on_test_epoch_start(self):
        self.test_results: list[Result] = []
        self.idx = 0

    def test_step(self, batch, _):
        strain, parameters = batch

        samples = self.model.sample(
            self.hparams.samples_per_event, context=strain
        )
        descaled = self.trainer.datamodule.scale(samples, reverse=True)
        parameters = self.trainer.datamodule.scale(parameters, reverse=True)

        result = self.cast_as_bilby_result(
            descaled.cpu().numpy(),
            parameters.cpu().numpy()[0],
        )
        self.test_results.append(result)

        # plot corner and skymap for a subset of the test results
        if self.idx < self.num_corner:
            skymap_filename = self.outdir / f"{self.idx}_mollview.png"
            corner_filename = self.outdir / f"{self.idx}_corner.png"
            result.plot_corner(
                save=True,
                filename=corner_filename,
                levels=(0.5, 0.9),
            )
            result.plot_mollview(
                outpath=skymap_filename,
            )
        self.idx += 1

    def on_test_epoch_end(self):
        # pp plot
        bilby.result.make_pp_plot(
            self.test_results,
            save=True,
            filename=self.outdir / "pp-plot.png",
            keys=self.inference_params,
        )

        # searched area cum hist
        searched_areas = []
        for result in self.test_results:
            searched_area = result.calculate_searched_area()
            searched_areas.append(searched_area)
        searched_areas = np.sort(searched_areas)
        counts = np.arange(1, len(searched_areas) + 1) / len(searched_areas)

        plt.figure(figsize=(10, 6))
        plt.step(searched_areas, counts, where="post")
        plt.xscale("log")
        plt.xlabel("Searched Area (deg^2)")
        plt.ylabel("Cumulative Probability")
        plt.title("Searched Area Cumulative Distribution Function")
        plt.savefig(self.outdir / "searched_area.png")
        np.save(self.outdir / "searched_area.npy", searched_areas)
