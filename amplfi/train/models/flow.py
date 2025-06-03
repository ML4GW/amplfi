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
    PlotMollview,
    PlotCorner,
    SaveFITS,
    SaveInjectionParameters,
)
from ...utils.result import AmplfiResult
from .base import AmplfiModel
from typing import Optional
from scipy.special import logsumexp

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
        filter_params:
            If `True`, filter the samples produced by the flow
            to keep only valid samples within the prior boundaries.
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
        plot_corner:
            If `True`, plot corner plots for
            testing set events
        plot_mollview:
            If `True`, plot mollview plots for
            testing set events
        cross_match:
            If `True`, run ligo.skymap.postprocess.crossmatch
            on result objects at the end of testing epoch
            and produce searched area and volume cdfs
        save_fits:
            If `True`, save skymaps as FITS files
            for testing set events
        save_posterior:
            If `True`, save bilby Result objects and posterior samples
        save_injection_parameters:
            If `True`, save the injection parameters for each event
            in the testing set to a an hdf5 file. Useful for
            testing datasets where the injection parameters are randomly
            sampled.
    """

    def __init__(
        self,
        *args,
        arch: FlowArchitecture,
        filter_params: bool = True,
        samples_per_event: int = 10000,
        nside: int = 32,
        min_samples_per_pix: int = 5,
        num_plot: int = 10,
        plot_data: bool = False,
        plot_corner: bool = True,
        plot_mollview: bool = True,
        cross_match: bool = True,
        save_fits: bool = True,
        save_posterior: bool = False,
        save_injection_parameters: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = arch
        self.samples_per_event = samples_per_event
        self.num_plot = num_plot
        self.nside = nside
        self.min_samples_per_pix = min_samples_per_pix
        self.plot_data = plot_data
        self.plot_corner = plot_corner
        self.plot_mollview = plot_mollview
        self.save_fits = save_fits
        self.save_posterior = save_posterior
        self.cross_match = cross_match
        self.save_injection_parameters = save_injection_parameters
        self.filter_params = filter_params

        # save our hyperparameters
        self.save_hyperparameters(ignore=["arch"])

    def forward(self, context, parameters) -> Tensor:
        return -self.model.log_prob(parameters, context=context)

    def training_step(self, batch, _):
        strain, asds, parameters, _ = batch
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
        strain, asds, parameters, _ = batch
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

    def on_test_epoch_start(self):
        self.test_results: list[AmplfiResult] = []

    def on_test_batch_end(self, outputs, *_):
        self.test_results.append(outputs)

    def test_step(self, batch, _) -> AmplfiResult:
        strain, asds, parameters, snr = batch
        context = (strain, asds)

        samples = self.model.sample(
            self.hparams.samples_per_event, context=context
        )
        log_probs = self.model.log_prob(samples, context)

        samples = samples.squeeze(1)
        log_probs = log_probs.squeeze(1)

        descaled = self.scale(samples, reverse=True)
        if self.filter_params:
            descaled, mask = self.filter_parameters(descaled)
            log_probs = log_probs[mask]

        parameters = self.scale(parameters, reverse=True)
        parameters = parameters.cpu().numpy()[0]

        # create a dictionary of injection parameters
        # mapping from parameter string to the true injection value
        # and add snr if provided
        injection_parameters = {
            k: float(v)
            for k, v in zip(self.inference_params, parameters, strict=False)
        }
        injection_parameters["ra"] = injection_parameters["phi"]

        if snr is not None:
            injection_parameters["snr"] = snr[0].item()

        result = self.cast_as_bilby_result(
            descaled.cpu().numpy(),
            log_probs.cpu().numpy(),
            injection_parameters,
        )

        return result

    def predict_step(self, batch, _):
        strain, asds, _ = batch
        context = (strain, asds)

        samples = self.model.sample(
            self.hparams.samples_per_event, context=context
        )
        log_probs = self.model.log_prob(samples, context)
        log_probs = log_probs.squeeze(1)
        samples = samples.squeeze(1)
        descaled = self.scale(samples, reverse=True)
        descaled, mask = self.filter_parameters(descaled)

        log_probs = log_probs[mask]
        result = self.cast_as_bilby_result(
            descaled.cpu().numpy(),
            log_probs.cpu().numpy(),
            None,
        )

        return result

    def cast_as_bilby_result(
        self,
        samples: np.ndarray,
        log_probs: np.ndarray,
        injection_parameters: Optional[dict[str, float]] = None,
    ) -> AmplfiResult:
        """Cast posterior samples as Bilby Result object
        for ease of producing corner and pp plots

        Args:
            samples:
                An array of posterior samples of shape
                (1, num_samples, num_params)
            injection_parameters:
                For injections, a dictionary mapping from parameter string
                to the true injection value

        """

        # create dummy prior with correct attributes
        # for making our results compatible with bilbys make_pp_plot
        priors = {
            param: bilby.core.prior.base.Prior(latex_label=param)
            for param in self.inference_params
        }
        posterior = {}
        for idx, k in enumerate(self.inference_params):
            posterior[k] = samples.T[idx].flatten()

        posterior["log_prob"] = log_probs
        posterior = pd.DataFrame(posterior)

        num_samples = len(posterior)
        log_evidence = logsumexp(posterior["log_prob"]) - np.log(num_samples)

        r = AmplfiResult(
            label="PEModel",
            injection_parameters=injection_parameters,
            posterior=posterior,
            search_parameter_keys=self.inference_params,
            priors=priors,
            log_evidence=log_evidence,
        )
        r.posterior["ra"] = r.posterior["phi"]
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

        return parameters, net_mask

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        callbacks += [ProbProbPlot()]
        if self.plot_data:
            callbacks.append(
                StrainVisualization(self.test_outdir / "events", self.num_plot)
            )

        if self.save_injection_parameters:
            callbacks.append(SaveInjectionParameters(self.test_outdir))

        event_outdir = self.test_outdir / "events"
        event_outdir.mkdir(parents=True, exist_ok=True)

        if self.save_fits:
            callbacks.append(
                SaveFITS(event_outdir, self.nside, self.min_samples_per_pix)
            )

        if self.plot_mollview:
            callbacks.append(PlotMollview(event_outdir, self.nside))

        if self.plot_corner:
            callbacks.append(PlotCorner(event_outdir))

        if self.save_posterior:
            callbacks.append(SavePosterior(event_outdir))

        if self.cross_match:
            callbacks.append(CrossMatchStatistics())

        return callbacks
