import bilby
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gwpy.plot import Plot
from gwpy.timeseries import TimeSeries

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
        samples_per_event: int = 20000,
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

    def test_step(self, batch, _):
        strain, asds, parameters = batch
        context = (strain, asds)

        samples = self.model.sample(
            self.hparams.samples_per_event, context=context
        )
        descaled = self.trainer.datamodule.scale(samples, reverse=True)
        parameters = self.trainer.datamodule.scale(parameters, reverse=True)

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
            strain_filename = self.outdir / f"{self.idx}_whitened.png"
            spec_filename = self.outdir / f"{self.idx}_spectrogram.png"
            asd_filename = self.outdir / f"{self.idx}_asd.png"

            plt.figure()
            plt.plot(strain.cpu().numpy()[0][0], label="H1")
            plt.plot(strain.cpu().numpy()[0][1], label="L1")
            plt.legend()
            plt.savefig(strain_filename)
            plt.close()

            qscans = []
            for i in range(2):
                ts = TimeSeries(
                    strain.cpu().numpy()[0][i],
                    dt=1 / self.trainer.datamodule.hparams.sample_rate,
                )
                spec = ts.q_transform(
                    logf=True, whiten=False, frange=(20, 512), qrange=(11, 11)
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

            plt.figure()
            plt.loglog(asds.cpu().numpy()[0][0], label="H1")
            plt.loglog(asds.cpu().numpy()[0][1], label="L1")

            plt.legend()
            plt.savefig(asd_filename)
            plt.close()

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
