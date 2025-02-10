import bilby
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..architectures.flows import FlowArchitecture
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
        samples_per_event:
            Number of samples to draw per event for testing
        nside:
            nside parameter for healpy
    """

    def __init__(
        self,
        *args,
        arch: FlowArchitecture,
        samples_per_event: int = 200000,
        num_corner: int = 10,
        nside: int = 32,
        min_samples_per_pix: int = 15,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        # construct our model
        self.model = arch
        self.samples_per_event = samples_per_event
        self.num_corner = num_corner
        self.nside = nside
        self.min_samples_per_pix = min_samples_per_pix

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

    def _get_log_prior_dict(self, trainer=None):
        """
        Get the log prior for each parameter in the model.
        """
        if trainer is not None:
            self.trainer = trainer
        log_prior_dict = {}
        sf = self.trainer.datamodule.waveform_sampler.parameter_sampler # need this to get the prior for some parameters

        for param in self.inference_params:
            if param in sf.parameters:
                # some priors live in the ```waveform_sampler.parameter_sampler``` object
                # {chirp_mass': Uniform(low: 10.0, high: 100.0),
                # 'mass_ratio': Uniform(low: 0.125, high: 0.9990000128746033),
                # 'distance': PowerLaw(),
                # 'inclination': Sine(),
                # 'phic': Uniform(low: 0.0, high: 6.2831854820251465),
                # 'chi1': DeltaFunction(),
                # 'chi2': DeltaFunction()}
                log_prior_dict[param] = sf.parameters[param]
            else:
                # others are seperately
                # waveform_sampler."name"
                # 'dec': Cosine(),
                # 'psi': Uniform(low: 0.0, high: 3.140000104904175),
                # 'phi': Uniform(low: -3.140000104904175, high: 3.140000104904175)
                log_prior_dict[param] = getattr(self.trainer.datamodule.waveform_sampler, param)
        # add low and high boundaries (for some parameters these values are missing)
        log_prior_dict['distance'].low = 100
        log_prior_dict['distance'].high = 3100
        log_prior_dict['inclination'].low = 0
        log_prior_dict['inclination'].high = np.pi
        log_prior_dict['dec'].low = -np.pi/2
        log_prior_dict['dec'].high = np.pi/2
        return log_prior_dict

    def filter_descaled_parameters(self, descaled):
        """
        Filter the descaled parameters to keep only valid samples within their boundaries.

        Args:
            descaled (torch.Tensor): The descaled parameters tensor.
        Returns:
            torch.Tensor: The filtered descaled parameters.
        """
        valid_idxs = torch.ones(descaled.shape[0], dtype=torch.bool, device=descaled.device)
        num_discarded = 0
        discarded_count = 0
        previous_discarded_count = 0
        for idx, param in enumerate(self.inference_params):
            prior = log_prior_dict[param]
            low = prior.low
            high = prior.high
            valid_idxs &= (descaled[:, idx] >= low) & (descaled[:, idx] <= high)
            discarded_count = (~valid_idxs).sum().item()
            self._logger.info(f"Discarded samples[{param}]: {discarded_count-previous_discarded_count}")
            num_discarded += discarded_count-previous_discarded_count
            previous_discarded_count = discarded_count
        self._logger.info(f"Total discarded samples: {num_discarded}/{descaled.shape[0]}")
        # Filter the descaled parameters to keep only valid samples
        descaled = descaled[valid_idxs]
        return descaled

    def test_step(self, batch, _):
        strain, asds, parameters = batch
        context = (strain, asds)

        samples = self.model.sample(
            self.hparams.samples_per_event, context=context
        )
        descaled = self.trainer.datamodule.scale(samples, reverse=True)
        parameters = self.trainer.datamodule.scale(parameters, reverse=True)
        # create a dictionary of prior objects for each parameter
        self.log_prior_dict = self._get_log_prior_dict()
        # filter out samples outside of prior boundaries
        descaled = self.filter_descaled_parameters(descaled)

        result = self.cast_as_bilby_result(
            descaled.cpu().numpy(),
            parameters.cpu().numpy()[0],
        )
        result.calculate_skymap(self.nside, self.min_samples_per_pix)
        self.test_results.append(result)

        # plot corner and skymap for a subset of the test results
        if self.idx < self.num_corner:
            skymap_filename = self.outdir / f"{self.idx}_mollview.png"
            corner_filename = self.outdir / f"{self.idx}_corner.png"
            fits_filename = self.outdir / f"{self.idx}.fits"
            result.plot_corner(
                save=True,
                filename=corner_filename,
                levels=(0.5, 0.9),
            )
            result.plot_mollview(
                outpath=skymap_filename,
            )
            result.fits_table.writeto(fits_filename, overwrite=True)
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
        fifty_percent_areas = []
        ninety_percent_areas = []
        for result in self.test_results:
            searched_area, fifty, ninety = result.calculate_searched_area(
                self.nside
            )
            searched_areas.append(searched_area)
            fifty_percent_areas.append(fifty)
            ninety_percent_areas.append(ninety)
        searched_areas = np.sort(searched_areas)
        counts = np.arange(1, len(searched_areas) + 1) / len(searched_areas)

        plt.figure(figsize=(10, 6))
        plt.step(searched_areas, counts, where="post")
        plt.xscale("log")
        plt.xlabel("Searched Area (deg^2)")
        plt.ylabel("Cumulative Probability")
        plt.title("Searched Area Cumulative Distribution Function")
        plt.grid()
        plt.axhline(0.5, color="grey", linestyle="--")
        plt.savefig(self.outdir / "searched_area.png")
        np.save(self.outdir / "searched_area.npy", searched_areas)

        plt.close()
        plt.figure(figsize=(10, 6))
        mm, bb, pp = plt.hist(
            fifty_percent_areas, label="50 percent area", bins=50
        )
        _, _, _ = plt.hist(
            ninety_percent_areas, label="90 percent area", bins=bb
        )
        plt.xlabel("Sq. deg.")
        plt.legend()
        plt.savefig(self.outdir / "fifty_ninety_areas.png")
