import bilby
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gwpy.plot import Plot
from gwpy.timeseries import TimeSeries

from ..architectures.flows import FlowArchitecture
from ..callbacks import StrainVisualization
from ..testing import Result
from .base import AmplfiModel

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
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = arch
        self.samples_per_event = samples_per_event
        self.num_plot = num_plot
        self.nside = nside
        self.min_samples_per_pix = min_samples_per_pix
        self.plot_data = plot_data

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
        posterior = {}
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

    def plot(
        self, strain: np.ndarray, asds: np.ndarray, result: bilby.result.Result
    ):
        """
        Create various plots for debugging purposes
        """
        sample_rate = self.trainer.datamodule.hparams.sample_rate

        strain_filename = self.outdir / f"{self.idx}_whitened.png"
        spec_filename = self.outdir / f"{self.idx}_spectrogram.png"
        asd_filename = self.outdir / f"{self.idx}_asd.png"
        freq_data_filename = self.outdir / f"{self.idx}_whitened_frequency.png"

        ifos = self.trainer.datamodule.hparams.ifos

        # whitened time domain strain
        plt.figure()
        plt.title("Whitened Time Domain Strain")

        for i, ifo in enumerate(ifos):
            plt.plot(strain[i], label=ifo)

        plt.legend()
        plt.savefig(strain_filename)
        plt.close()

        # qscans
        qscans = []

        for i in range(len(ifos)):
            ts = TimeSeries(
                strain[i],
                dt=1 / sample_rate,
            )

            spec = ts.q_transform(
                logf=True,
                whiten=False,
                frange=(25, 200),
                qrange=(4, 108),
                outseg=(3.35, 3.6),
            )
            qscans.append(spec)

        chirp_mass, mass_ratio = (
            result.injection_parameters["chirp_mass"],
            result.injection_parameters["mass_ratio"],
        )
        title = f"chirp_mass: {chirp_mass:2f}, mass_ratio: {mass_ratio:2f}"
        plot = Plot(
            *qscans,
            figsize=(18, 5),
            geometry=(1, 2),
            yscale="log",
            method="pcolormesh",
            cmap="viridis",
            title=title,
        )

        for i, ax in enumerate(plot.axes):
            label = "" if i != 1 else "Normalized Energy"
            plot.colorbar(ax=ax, label=label)

        plot.savefig(spec_filename)
        plt.close()

        # asds
        frequencies = self.trainer.datamodule.frequencies.numpy()
        mask = frequencies > self.trainer.datamodule.hparams.highpass
        frequencies_masked = frequencies[mask]
        plt.figure()
        for i, ifo in enumerate(ifos):
            plt.loglog(frequencies_masked, asds[i], label=f"{ifo} asd")
            plt.title("ASDs")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Scaled Amplitude")

        plt.legend()
        plt.savefig(asd_filename)
        plt.close()

        # whitened frequency domain data
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for i, ifo in enumerate(ifos):
            strain_fft = np.fft.rfft(strain[i])
            freqs = np.fft.rfftfreq(n=strain[i].shape[-1], d=1 / sample_rate)

            axes[0].plot(
                freqs,
                strain_fft.real / (sample_rate ** (1 / 2)),
                label=f"{ifo} data",
            )
            axes[1].plot(
                freqs,
                strain_fft.imag / (sample_rate ** (1 / 2)),
                label=f"{ifo} data",
            )

        axes[0].set_title("Real")
        axes[1].set_title("Imaginary")

        axes[0].set_ylabel("Whitened Amplitude")
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].set_xlabel("Frequency (Hz)")

        plt.legend()

        plt.savefig(freq_data_filename)
        plt.close()

    def on_test_epoch_start(self):
        self.test_results: list[Result] = []
        self.idx = 0

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

    def test_step(self, batch, _) -> bilby.result.Result:
        strain, asds, parameters = batch
        context = (strain, asds.clone())

        samples = self.model.sample(
            self.hparams.samples_per_event, context=context
        )
        descaled = self.trainer.datamodule.scale(samples, reverse=True)
        descaled = self.filter_parameters(descaled)
        parameters = self.trainer.datamodule.scale(parameters, reverse=True)

        result = self.cast_as_bilby_result(
            descaled.cpu().numpy(),
            parameters.cpu().numpy()[0],
        )
        result.calculate_skymap(self.nside, self.min_samples_per_pix)
        self.test_results.append(result)

        # plot corner and skymap for a subset of the test results
        if self.idx < self.num_plot:
            skymap_filename = self.test_outdir / f"{self.idx}_mollview.png"
            corner_filename = self.test_outdir / f"{self.idx}_corner.png"
            fits_filename = self.test_outdir / f"{self.idx}.fits"
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

        return result

    def on_test_epoch_end(self):
        # pp plot
        bilby.result.make_pp_plot(
            self.test_results,
            save=True,
            filename=self.test_outdir / "pp-plot.png",
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
        plt.savefig(self.test_outdir / "searched_area.png")
        np.save(self.test_outdir / "searched_area.npy", searched_areas)

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
        plt.savefig(self.test_outdir / "fifty_ninety_areas.png")

    def configure_callbacks(self):
        callbacks = []
        if self.plot_data:
            callbacks.append(
                StrainVisualization(self.test_outdir, self.num_plot)
            )
        return callbacks
