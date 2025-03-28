import bilby
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import torch

from ..architectures.flows import FlowArchitecture
from ..callbacks import (
    StrainVisualization,
    SavePosterior,
    CrossMatchStatistics,
    ProbProbPlot,
)
from ...utils.result import AmplfiResult
from .base import AmplfiModel
from typing import Optional

if TYPE_CHECKING:
    pass

Tensor = torch.Tensor


class FlowModel(AmplfiModel):
    """
    A LightningModule for training normalizing flows

    Args:
        *args:
            See arguments in `amplfi.train.models.base.AmplfiModel`
        arch:
            Neural network architecture to train.
            This should be a subclass of `FlowArchitecture`.
        samples_per_event:
            Number of samples to draw per event for testing
        nside:
            Healpix nside used when creating skymaps
        min_samples_per_pix:
        num_plot:
            Number of testing events to plot skymaps, corner
            plots and, if `plot_data` is `True`, strain data
            visualizations.
        plot_data:
            If `True`, plot strain visualization for `num_plot`
            of the testing set events
        save_posterior:
            If `True`, save bilby Result objects and posterior samples
        cross_match:
            If `True`, run ligo.skymap.postprocess.crossmatch
            on result objects at the end of testing epoch
            and produce searched area and volume cdfs
    """

    def __init__(
        self,
        *args,
        arch: FlowArchitecture,
        samples_per_event: int = 10000,
        nside: int = 32,
        min_samples_per_pix: int = 15,
        num_plot: int = 10,
        plot_data: bool = False,
        save_posterior: bool = False,
        cross_match: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = arch
        self.samples_per_event = samples_per_event
        self.num_plot = num_plot
        self.nside = nside
        self.min_samples_per_pix = min_samples_per_pix
        self.plot_data = plot_data
        self.save_posterior = save_posterior
        self.cross_match = cross_match

        # save our hyperparameters
        self.save_hyperparameters(ignore=["arch"])

    def forward(self, context, parameters) -> Tensor:
        return -self.model.log_prob(parameters, context=context)

    def training_step(self, batch, _):
        strain, asds, parameters = batch
        context = (strain, asds)
        loss = self(context, parameters).mean()
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
        strain, asds, parameters = batch
        context = (strain, asds)
        loss = self(context, parameters).mean()
        self.log(
            "valid_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx) -> AmplfiResult:
        strain, asds, parameters = batch
        context = (strain, asds)

        samples = self.model.sample(
            self.hparams.samples_per_event, context=context
        )
        samples = samples.squeeze(1)
        descaled = self.scale(samples, reverse=True)
        descaled = self.filter_parameters(descaled)
        parameters = self.scale(parameters, reverse=True)

        result = self.cast_as_bilby_result(
            descaled.cpu().numpy(),
            parameters.cpu().numpy()[0],
        )

        test_outdir = self.test_outdir / f"event_{batch_idx}"
        test_outdir.mkdir(parents=True, exist_ok=True)

        # add ra column for use with ligo-skymap-from-samples
        result.posterior["ra"] = result.posterior["phi"]

        self.test_results.append(result)

        # plot corner and skymap
        skymap_filename = test_outdir / "mollview.png"
        corner_filename = test_outdir / "corner.png"
        # fits_filename = test_outdir / "amplfi.skymap.fits"
        result.plot_corner(
            save=True,
            filename=corner_filename,
            levels=(0.5, 0.9),
        )
        result.plot_mollview(
            outpath=skymap_filename,
        )

        return result

    def predict_step(self, batch, batch_idx, _):
        strain, asds, gpstime = batch
        context = (strain, asds)

        samples = self.model.sample(
            self.hparams.samples_per_event, context=context
        )
        descaled = self.scale(samples, reverse=True)
        descaled = self.filter_parameters(descaled)
        result = self.cast_as_bilby_result(
            descaled.cpu().numpy(),
            None,
        )

        test_outdir = self.test_outdir / f"event_{batch_idx}"
        test_outdir.mkdir(parents=True, exist_ok=True)

        skymap_filename = test_outdir / f"{gpstime.item()}_mollview.png"
        corner_filename = test_outdir / f"{gpstime.item()}_corner.png"
        # fits_filename = test_outdir / f"{gpstime.item()}.fits"

        result.plot_corner(
            save=True,
            filename=corner_filename,
            levels=(0.5, 0.9),
        )
        result.plot_mollview(
            outpath=skymap_filename,
        )

        return result

    def on_test_epoch_start(self):
        self.test_results: list[AmplfiResult] = []

    def cast_as_bilby_result(
        self,
        samples: np.ndarray,
        truth: Optional[np.ndarray] = None,
    ) -> AmplfiResult:
        """Cast posterior samples as Bilby Result object
        for ease of producing corner and pp plots

        Args:
            samples: posterior samples (1, num_samples, num_params)
            truth: true values of the parameters  (1, num_params)

        """

        injection_parameters = (
            {
                k: float(v)
                for k, v in zip(self.inference_params, truth, strict=False)
            }
            if truth is not None
            else None
        )

        # create dummy prior with correct attributes
        # for making our results compatible with bilbys make_pp_plot
        priors = {
            param: bilby.core.prior.base.Prior(latex_label=param)
            for param in self.inference_params
        }
        posterior = {}
        for idx, k in enumerate(self.inference_params):
            posterior[k] = samples.T[idx].flatten()
        posterior = pd.DataFrame(posterior)

        r = AmplfiResult(
            label="PEModel",
            injection_parameters=injection_parameters,
            posterior=posterior,
            search_parameter_keys=self.inference_params,
            priors=priors,
        )
        return r

    def filter_parameters(self, parameters: torch.Tensor):
        """
        Filter the descaled parameters to keep only valid samples
        within their boundaries.

        Args:
            descaled (torch.Tensor): The descaled parameters tensor.

        Returns:
            torch.Tensor: The filtered descaled parameters.
        """
        net_mask = torch.ones(
            parameters.shape[0], dtype=bool, device=parameters.device
        )
        waveform_sampler = self.trainer.datamodule.waveform_sampler
        priors = waveform_sampler.parameter_sampler.parameters
        for i, param in enumerate(self.inference_params):
            samples = parameters[:, i]
            if param in ["dec", "phi", "psi"]:
                prior = getattr(
                    self.trainer.datamodule.waveform_sampler, param
                )
            else:
                prior = priors[param]

            mask = (prior.log_prob(samples) == float("-inf")).to(
                samples.device
            )
            self._logger.debug(
                f"Removed {mask.sum()}/{len(mask)} samples for parameter "
                f"{param} outside of prior range"
            )

            net_mask &= ~mask

        self._logger.info(
            f"Removed {(~net_mask).sum()}/{len(net_mask)} total samples "
            "outside of prior range"
        )
        parameters = parameters[net_mask]

        return parameters

    def configure_callbacks(self):
        callbacks = [ProbProbPlot()]
        if self.plot_data:
            callbacks.append(
                StrainVisualization(self.test_outdir, self.num_plot)
            )

        if self.save_posterior:
            callbacks.append(SavePosterior(self.test_outdir))

        if self.cross_match:
            callbacks.append(CrossMatchStatistics())

        return callbacks
