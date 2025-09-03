import bilby
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import torch
from ..architectures.flows import FlowArchitecture
from ...utils.result import AmplfiResult
from .base import AmplfiModel
from typing import Optional
from bilby.core.prior import PriorDict
from ..data.datasets.testing import ra_from_phi
from amplfi.train.callbacks import (
    ProbProbPlot,
    CrossMatchStatistics,
    SaveInjectionParameters,
)

if TYPE_CHECKING:
    from pathlib import Path
    from ..prior import AmplfiPrior

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
        min_samples_per_pix_dist:
        max_samples_per_pixel:
        crossmatch:
            If `True`, run ligo.skymap.postprocess.crossmatch
            on result objects at the end of testing epoch
            and produce searched area and volume cdfs
        target_prior:
            Path to a bilby prior file for reweighting posterior samples to
            a new prior.
    """

    def __init__(
        self,
        *args,
        arch: FlowArchitecture,
        filter_params: bool = True,
        samples_per_event: int = 10000,
        min_samples_per_pix_dist: int = 5,
        max_samples_per_pixel: int = 20,
        target_prior: Optional["Path"] = None,
        crossmatch: bool = True,
        create_pp_plot: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = arch
        self.samples_per_event = samples_per_event
        self.min_samples_per_pix_dist = min_samples_per_pix_dist
        self.max_samples_per_pixel = max_samples_per_pixel
        self.filter_params = filter_params
        self.crossmatch = crossmatch
        self.create_pp_plot = create_pp_plot
        if target_prior is not None:
            target_prior = PriorDict(filename=str(target_prior))
        self.target_prior: Optional[PriorDict] = target_prior

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

    def on_predict_epoch_start(self):
        self.on_test_epoch_start()

    def on_test_epoch_start(self):
        self.test_outdir.mkdir(exist_ok=True, parents=True)
        (self.test_outdir / "events").mkdir(exist_ok=True, parents=True)
        self.test_results: list[AmplfiResult] = []
        self.reweighted_results: list[AmplfiResult] = []

        # update the training prior to now include
        # the extrinisc parameters, so log probabilites can be calculated
        waveform_sampler = self.trainer.datamodule.waveform_sampler
        training_prior = waveform_sampler.training_prior

        for key in ["dec", "psi", "phi"]:
            training_prior.priors[key] = getattr(self.trainer.datamodule, key)

        self.training_prior: "AmplfiPrior" = training_prior

        # if reweighting, write target prior to test directory
        if self.target_prior is not None:
            self.target_prior.to_file(self.test_outdir, label="reweight")
            (self.test_outdir / "reweighted").mkdir(exist_ok=True)

    def on_test_batch_end(self, outputs, *_):
        result: "AmplfiResult"
        reweighted: Optional["AmplfiResult"]
        result, reweighted = outputs
        self.test_results.append(result)

        if reweighted is not None:
            self.reweighted_results.append(reweighted)

    def analyze_event(
        self, strain, asds, parameters=None, snr=None, gpstime=None
    ) -> tuple[AmplfiResult, Optional[AmplfiResult]]:
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

        # convert samples to dictionary for
        # calculating log probabilites
        samples_dict = dict(
            zip(
                self.hparams.inference_params,
                descaled.transpose(1, 0),
                strict=True,
            )
        )

        # calculate training prior probability of posterior samples
        log_prior_of_posterior_samples = self.training_prior.log_prob(
            samples_dict
        )

        # when predicting, there will be no ground truth parameters
        injection_parameters = None
        if parameters is not None:
            parameters = self.scale(parameters, reverse=True)
            parameters = parameters.cpu().numpy()[0]

            # create a dictionary of injection parameters
            # mapping from parameter string to the true
            # injection value, and add snr if provided
            injection_parameters = {
                k: float(v)
                for k, v in zip(
                    self.inference_params, parameters, strict=False
                )
            }
            injection_parameters["ra"] = injection_parameters["phi"]

            if snr is not None:
                injection_parameters["snr"] = snr[0].item()

        # when predicting on real strain, convert
        # phi to the physical ra value
        if gpstime is not None:
            phi_idx = self.hparams.inference_params.index("phi")
            phis = descaled[:, phi_idx]
            ras = ra_from_phi(phis, gpstime)
            descaled[:, phi_idx] = ras

        result = self.cast_as_bilby_result(
            descaled.cpu().numpy(),
            log_probs.cpu().numpy(),
            log_prior_of_posterior_samples.cpu().numpy(),
            injection_parameters,
        )

        # optionally reweight to different prior
        # TODO: include likelihood reweighting
        reweighted_result: Optional[AmplfiResult] = None
        if self.target_prior is not None:
            self._logger.info(
                f"Reweighting {len(result.posterior)} posterior samples"
            )
            reweighted_result = result.reweight_to_prior(self.target_prior)
            self._logger.info(
                f"{len(reweighted_result.posterior)} posterior samples "
                "after rejection sampling"
            )

        return result, reweighted_result

    def test_step(
        self, batch, _
    ) -> tuple[AmplfiResult, Optional[AmplfiResult]]:
        strain, asds, parameters, snr = batch
        return self.analyze_event(strain, asds, parameters, snr)

    def predict_step(
        self, batch, _
    ) -> tuple[AmplfiResult, Optional[AmplfiResult]]:
        strain, asds, gpstime = batch
        return self.analyze_event(strain, asds, None, None, gpstime[0].cpu())

    def cast_as_bilby_result(
        self,
        samples: np.ndarray,
        log_probs: np.ndarray,
        log_prior_probs: np.ndarray,
        injection_parameters: Optional[dict[str, float]] = None,
    ) -> AmplfiResult:
        """Cast posterior samples as Bilby Result object
        for ease of producing corner and pp plots

        Args:
            samples:
                An array of posterior samples of shape
                (1, num_samples, num_params)
            log_probs:
                An array of log probabilities of posterior samples
                as predicted under the normalizing flow model
            log_prior_probs:
                An array of log prior probabilities of posterior samples
                as predicted under the training prior
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

        r = AmplfiResult(
            label="PEModel",
            injection_parameters=injection_parameters,
            posterior=posterior,
            search_parameter_keys=self.inference_params,
            priors=priors,
            log_prior_evaluations=log_prior_probs,
        )
        r.posterior["ra"] = r.posterior["phi"]
        r.posterior["log_prior"] = log_prior_probs
        return r

    def filter_parameters(
        self, parameters: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        priors = self.training_prior.priors
        for i, param in enumerate(self.inference_params):
            samples = parameters[:, i]
            if param in ["dec", "psi", "phi"]:
                prior = getattr(self.trainer.datamodule, param)
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
        # automatically add since now
        # supported by all test datasets
        callbacks += [SaveInjectionParameters()]
        if self.crossmatch:
            callbacks.append(
                CrossMatchStatistics(
                    self.min_samples_per_pix_dist, self.max_samples_per_pixel
                )
            )
        if self.create_pp_plot:
            callbacks.append(ProbProbPlot())
        return callbacks
