import logging
import torch
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from gwpy.plot import Plot
from gwpy.timeseries import TimeSeries
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import bilby
from amplfi.utils.skymap import plot_skymap
from tqdm.auto import tqdm
import ligo.skymap.plot  # noqa: F401
from ligo.skymap.io.fits import write_sky_map
import pandas as pd

if TYPE_CHECKING:
    from ligo.skymap.postprocess.crossmatch import CrossmatchResult
    from amplfi.train.models import FlowModel
    from amplfi.utils.result import AmplfiResult


class StrainVisualization(pl.Callback):
    """
    Lightning Callback for visualizing the strain data
    and asds being analyzed during the test step

    Args:
        save_data:
            Save the raw strain data to disk
        num_plot:
            Number of events to plot and (optionally) save data for.
            Defaults to all testing events
    """

    def __init__(self, save_data: bool = True, num_plot: Optional[int] = None):
        self.num_plot = num_plot or float("inf")
        self.save_data = save_data

    def plot_strain(self, outdir, result, batch, trainer):
        """
        Called at the end of each test step.
        `outputs` consists of objects returned by `pl_module.test_step`.
        """

        # unpack batch
        strain, asds, *_ = batch
        strain, asds = strain[0].cpu().numpy(), asds[0].cpu().numpy()

        # steal some attributes needed from datamodule
        # TODO: should we link these to the pl_module?
        highpass = trainer.datamodule.hparams.highpass
        sample_rate = trainer.datamodule.hparams.sample_rate
        ifos = trainer.datamodule.hparams.ifos

        # filenames for various plots
        whitened_td_strain_fname = outdir / "whitened_td.png"
        whitened_fd_strain_fname = outdir / "whitened_fd.png"
        qscan_fname = outdir / "qscan.png"
        asd_fname = outdir / "asd.png"

        # whitened time domain strain
        plt.figure()
        plt.title("Whitened Time Domain Strain")

        for i, ifo in enumerate(ifos):
            plt.plot(strain[i], label=ifo)
            if self.save_data:
                np.savetxt(outdir / "whitened_td_strain.txt", strain)

        plt.legend()
        plt.savefig(whitened_td_strain_fname)
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
                frange=(25, 500),
                qrange=(4, 108),
                # outseg=(3, 3.7),
            )
            qscans.append(spec)

        title = ""
        if result.injection_parameters is not None:
            chirp_mass, mass_ratio = (
                result.injection_parameters["chirp_mass"],
                result.injection_parameters["mass_ratio"],
            )
            title = f"chirp_mass: {chirp_mass:2f}, mass_ratio: {mass_ratio:2f}"
        plot = Plot(
            *qscans,
            figsize=(18, 5),
            geometry=(1, len(ifos)),
            yscale="log",
            method="pcolormesh",
            cmap="viridis",
            title=title,
        )

        for i, ax in enumerate(plot.axes):
            label = "" if i != 1 else "Normalized Energy"
            plot.colorbar(ax=ax, label=label)

        plot.savefig(qscan_fname)
        plt.close()

        # asds
        frequencies = trainer.datamodule.frequencies.numpy()
        mask = frequencies > highpass
        frequencies_masked = frequencies[mask]
        plt.figure()
        for i, ifo in enumerate(ifos):
            plt.loglog(frequencies_masked, asds[i], label=f"{ifo} asd")
            plt.title("ASDs")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Scaled Amplitude")

        if self.save_data:
            np.savetxt(
                outdir / "asds.txt",
                np.vstack((np.expand_dims(frequencies_masked, axis=0), asds)),
            )
        plt.legend()
        plt.savefig(asd_fname)
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

        axes[0].set_title("real")
        axes[1].set_title("imaginary")

        axes[0].set_ylabel("whitened amplitude")
        axes[0].set_xlabel("frequency (hz)")
        axes[0].set_xlabel("frequency (hz)")

        plt.legend()

        plt.savefig(whitened_fd_strain_fname)
        plt.close()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        outdir = pl_module.test_outdir / "events"
        # test_step returns bilby result object
        # and optionally a reweighted result
        result: "AmplfiResult"
        _: Optional["AmplfiResult"]
        result, _ = outputs

        if batch_idx >= self.num_plot:
            return

        outdir = outdir / str(batch_idx)
        outdir.mkdir(exist_ok=True)
        self.plot_strain(outdir, result, batch, trainer)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        outdir = pl_module.test_outdir / "events"
        # test_step returns bilby result object
        # and optionally a reweighted result
        result: "AmplfiResult"
        _: Optional["AmplfiResult"]
        result, _ = outputs

        if batch_idx >= self.num_plot:
            return

        gpstime = batch[2].cpu().numpy()[0]
        outdir = outdir / str(int(gpstime))
        outdir.mkdir(exist_ok=True)
        self.plot_strain(outdir, result, batch, trainer)


class PlotSkymap(pl.Callback):
    """
    Lightning Callback for plotting skymaps after each test batch

    Args:
        max_samples_per_pixel:
            See `ligo.skymap.healpix_tree.adaptive_healpix_histogram`
        min_samples_per_pix_dist:
            See `amplfi.train.skymap.histogram_skymap`
    """

    def __init__(
        self,
        max_samples_per_pixel: int = 20,
        min_samples_per_pix_dist: int = 5,
    ):
        self.max_samples_per_pixel = max_samples_per_pixel
        self.min_samples_per_pix_dist = min_samples_per_pix_dist

    def _plot_skymap(
        self,
        outdir: Path,
        result: "AmplfiResult",
        reweighted: Optional["AmplfiResult"] = None,
    ):
        skymap = result.to_skymap(
            max_samples_per_pixel=self.max_samples_per_pixel,
            min_samples_per_pix_dist=self.min_samples_per_pix_dist,
        )
        ra = result.injection_parameters["ra"]
        dec = result.injection_parameters["dec"]

        plot_skymap(skymap, ra, dec, outdir / "skymap.png")

        if reweighted is not None:
            reweighted_outdir = outdir / "reweighted"
            reweighted_outdir.mkdir(exist_ok=True)
            reweighted_skymap = result.to_skymap(
                max_samples_per_pixel=self.max_samples_per_pixel,
                min_samples_per_pix_dist=self.min_samples_per_pix_dist,
            )
            plot_skymap(
                reweighted_skymap, ra, dec, reweighted_outdir / "skymap.png"
            )

    def on_test_batch_end(
        self,
        trainer,
        pl_module: "FlowModel",
        outputs: "AmplfiResult",
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        """
        Called at the end of each test step.
        `outputs` consists of objects returned by `pl_module.test_step`.
        """

        outdir = pl_module.test_outdir / "events"
        # test_step returns bilby result object
        # and optionally a reweighted result
        result: "AmplfiResult"
        reweighted: Optional["AmplfiResult"]
        result, reweighted = outputs

        outdir = outdir / str(batch_idx)
        outdir.mkdir(exist_ok=True)
        self._plot_skymap(outdir, result, reweighted)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        outdir = pl_module.test_outdir / "events"
        # test_step returns bilby result object
        # and optionally a reweighted result
        result: "AmplfiResult"
        reweighted: Optional["AmplfiResult"]
        result, reweighted = outputs

        # TODO: remove "magic" index 2
        gpstime = batch[2].cpu().numpy()[0]
        outdir = outdir / str(int(gpstime))
        outdir.mkdir(exist_ok=True)
        self._plot_skymap(outdir, result, reweighted)


class PlotCorner(pl.Callback):
    """
    Lightning Callback for making corner plots after test epoch
    """

    def plot_corner(self, result: "AmplfiResult", outdir: Path):
        corner_filename = outdir / "corner.png"
        outdir.mkdir(exist_ok=True)
        result.plot_corner(
            save=True,
            filename=corner_filename,
            levels=(0.5, 0.9),
        )

    def on_test_batch_end(
        self,
        trainer,
        pl_module: "FlowModel",
        outputs: "AmplfiResult",
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        """
        Called at the end of each test step.
        `outputs` consists of objects returned by `pl_module.test_step`.
        """

        outdir = pl_module.test_outdir / "events"
        # test_step returns bilby result object
        # and optionally a reweighted result
        result: "AmplfiResult"
        reweighted: Optional["AmplfiResult"]
        result, reweighted = outputs

        outdir = outdir / str(batch_idx)
        self.plot_corner(result, outdir)

        if reweighted is not None:
            outdir = outdir / "reweighted"
            self.plot_corner(reweighted, outdir)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        outdir = pl_module.test_outdir / "events"
        # test_step returns bilby result object
        # and optionally a reweighted result
        result: "AmplfiResult"
        reweighted: Optional["AmplfiResult"]
        result, reweighted = outputs

        # for predict step, use gpstime to name the directory
        gpstime = batch[2].cpu().numpy()[0]
        outdir = outdir / str(int(gpstime))
        self.plot_corner(result, outdir)

        if reweighted is not None:
            # for predict step, use gpstime to name the directory
            outdir = outdir / "reweighted"
            self.plot_corner(reweighted, outdir)


class SaveFITS(pl.Callback):
    """
    Save skymap fits file for each test event
    """

    def __init__(
        self,
        min_samples_per_pix_dist: int = 5,
        max_samples_per_pixel: int = 20,
    ):
        self.min_samples_per_pix_dist = min_samples_per_pix_dist
        self.max_samples_per_pixel = max_samples_per_pixel

    def save_fits(
        self,
        result: "AmplfiResult",
        outdir: Path,
    ):
        skymap = result.to_skymap(
            use_distance=True,
            min_samples_per_pix_dist=self.min_samples_per_pix_dist,
            max_samples_per_pixel=self.max_samples_per_pixel,
        )
        outdir.mkdir(exist_ok=True)
        write_sky_map(outdir / "amplfi.skymap.fits", skymap)

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs: "AmplfiResult",
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        """
        Called at the end of each test step.
        `outputs` consists of objects returned by `pl_module.test_step`.
        """
        outdir = pl_module.test_outdir / "events"
        # test_step returns bilby result object
        # and optionally a reweighted result
        result: "AmplfiResult"
        reweighted: Optional["AmplfiResult"]
        result, reweighted = outputs

        outdir = outdir / str(batch_idx)
        outdir.mkdir(exist_ok=True)
        self.save_fits(result, outdir)

        if reweighted is not None:
            outdir = outdir / "reweighted"
            outdir.mkdir(exist_ok=True)
            self.save_fits(reweighted, outdir)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        outdir = pl_module.test_outdir / "events"
        result: "AmplfiResult"
        reweighted: Optional["AmplfiResult"]
        result, reweighted = outputs

        # for predict step, use gpstime to name the directory
        gpstime = batch[2].cpu().numpy()[0]
        outdir = outdir / str(int(gpstime))
        self.save_fits(result, outdir)

        if reweighted is not None:
            outdir = outdir / "reweighted"
            self.save_fits(reweighted, outdir)


class SavePosterior(pl.Callback):
    """
    Save bilby result objects and posterior csv data to disk
    for each test event
    """

    def save_posterior(
        self,
        result: "AmplfiResult",
        outdir: Path,
    ):
        # save posterior samples for ease of use with
        # ligo skymap and save full result to have
        # access to the true injection parameters
        result.save_posterior_samples(outdir / "posterior_samples.dat")
        result.save_to_file(
            outdir / "result.hdf5", extension="hdf5", overwrite=True
        )

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs: "AmplfiResult",
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        """
        Called at the end of each test step.
        `outputs` consists of objects returned by `pl_module.test_step`.
        """
        outdir = pl_module.test_outdir / "events"
        # test_step returns bilby result object
        # and optionally a reweighted result
        result: "AmplfiResult"
        reweighted: Optional["AmplfiResult"]
        result, reweighted = outputs

        outdir = outdir / str(batch_idx)
        outdir.mkdir(exist_ok=True)
        self.save_posterior(result, outdir)

        if reweighted is not None:
            outdir = outdir / "reweighted"
            outdir.mkdir(exist_ok=True)
            self.save_posterior(reweighted, outdir)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        outdir = pl_module.test_outdir / "events"
        # test_step returns bilby result object
        # and optionally a reweighted result
        result: "AmplfiResult"
        reweighted: Optional["AmplfiResult"]
        result, reweighted = outputs

        # for predict step, use gpstime to name the directory
        gpstime = batch[2].cpu().numpy()[0]

        outdir = outdir / str(int(gpstime))
        outdir.mkdir(exist_ok=True)
        self.save_posterior(result, outdir)

        if reweighted is not None:
            outdir = outdir / "reweighted"
            outdir.mkdir(exist_ok=True)
            self.save_posterior(reweighted, outdir)


class ProbProbPlot(pl.Callback):
    """
    Create a pp-plot of test events
    """

    def on_test_epoch_end(self, _, pl_module: "FlowModel"):
        logger = pl_module._logger
        logger.info("Making PP plot")
        bilby.result.make_pp_plot(
            pl_module.test_results,
            save=True,
            filename=pl_module.test_outdir / "plots" / "pp-plot.png",
            keys=pl_module.inference_params,
        )

        if pl_module.reweighted_results:
            outdir = pl_module.test_outdir / "reweighted" / "plots"
            outdir.mkdir(exist_ok=True, parents=True)
            bilby.result.make_pp_plot(
                pl_module.reweighted_results,
                save=True,
                filename=outdir / "pp-plot.png",
                keys=pl_module.inference_params,
            )


def crossmatch_skymap(
    result: "AmplfiResult",
    min_samples_per_pix_dist: int,
    max_samples_per_pixel: int,
    contours: tuple[float],
):
    """
    Function to process each skymap and calculate crossmatch statistics.
    """

    crossmatch_result = result.to_crossmatch_result(
        use_distance=True,
        min_samples_per_pix_dist=min_samples_per_pix_dist,
        max_samples_per_pixel=max_samples_per_pixel,
        contours=contours,
    )
    return crossmatch_result


class CrossMatchStatistics(pl.Callback):
    """
    Use `ligo.skymap.postprocess.crossmatch` to calculate skymap statistics.
    Make searched area and searched volume CDF plots

    Args:
       contours:
          List of percentiles to calculate contour areas for.
          See `ligo.skymap.postprocess.crossmatch`
    """

    crossmatch_attributes = [
        "searched_area",
        "searched_vol",
        "searched_prob",
        "searched_prob_vol",
        "searched_prob_dist",
        "offset",
        "contour_areas",
    ]

    def __init__(
        self,
        min_samples_per_pix_dist: int = 5,
        max_samples_per_pixel: int = 20,
        contours: tuple[float] = (0.5, 0.9),
    ):
        self.contours = contours
        self.min_samples_per_pix_dist = min_samples_per_pix_dist
        self.max_samples_per_pixel = max_samples_per_pixel

    def write_skymap_statistics(
        self,
        outdir: Path,
        results: list["CrossmatchResult"],
        index: pd.Index,
    ):
        outdir.mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame(index=index)
        for attr in self.crossmatch_attributes:
            if attr == "contour_areas":
                for i, contour in enumerate(self.contours):
                    data = [result.contour_areas[i] for result in results]
                    df[f"{attr}_{int(contour * 100)}"] = data
            else:
                df[attr] = [getattr(result, attr) for result in results]
        df.to_hdf(outdir / "skymap_stats.hdf5", key="stats", mode="w")

    def on_test_epoch_end(self, trainer, pl_module: "FlowModel"):
        self.crossmatch(
            pl_module.test_results,
            pl_module.test_outdir,
            self.min_samples_per_pix_dist,
            self.max_samples_per_pixel,
            trainer.datamodule.test_parameters.index,
        )

        if pl_module.reweighted_results:
            self.crossmatch(
                pl_module.reweighted_results,
                pl_module.test_outdir / "reweighted",
                self.min_samples_per_pix_dist,
                self.max_samples_per_pixel,
                trainer.datamodule.test_parameters.index,
            )

    def crossmatch(
        self,
        results: list["AmplfiResult"],
        outdir: Path,
        min_samples_per_pix_dist: int,
        max_samples_per_pixel: int,
        index: pd.Index,
    ) -> None:
        (outdir / "plots").mkdir(exist_ok=True)
        func = partial(
            crossmatch_skymap,
            min_samples_per_pix_dist=min_samples_per_pix_dist,
            max_samples_per_pixel=max_samples_per_pixel,
            contours=self.contours,
        )

        crossmatch_results: list["CrossmatchResult"] = [None] * len(results)
        num_processes = min(mp.cpu_count(), len(results))
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_index = {
                executor.submit(func, result): idx
                for idx, result in enumerate(results)
            }
            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc="Crossmatching skymaps",
            ):
                idx = future_to_index[future]
                crossmatch_results[idx] = future.result()

        self.write_skymap_statistics(outdir, crossmatch_results, index)

        # searched area cum hist
        searched_areas = [
            result.searched_area for result in crossmatch_results
        ]
        searched_volumes = [
            result.searched_vol for result in crossmatch_results
        ]
        fifty_percent_areas = [
            result.contour_areas[0] for result in crossmatch_results
        ]
        ninety_percent_areas = [
            result.contour_areas[1] for result in crossmatch_results
        ]

        searched_areas = np.sort(searched_areas)
        searched_volumes = np.sort(searched_volumes)
        counts = np.arange(1, len(searched_areas) + 1) / len(searched_areas)

        # searched volume cum hist
        plt.figure(figsize=(10, 6))
        plt.step(searched_volumes, counts, where="post")
        plt.xscale("log")
        plt.xlabel("Searched Volume (Mpc^3)")
        plt.ylabel("Cumulative Probability")
        plt.title("Searched Volume Cumulative Distribution Function")
        plt.grid()
        plt.axhline(0.5, color="grey", linestyle="--")
        plt.savefig(outdir / "plots" / "searched_volume.png")

        # searched area cum hist
        plt.figure(figsize=(10, 6))
        plt.step(searched_areas, counts, where="post")
        plt.xscale("log")
        plt.xlabel("Searched Area (deg^2)")
        plt.ylabel("Cumulative Probability")
        plt.title("Searched Area Cumulative Distribution Function")
        plt.grid()
        plt.axhline(0.5, color="grey", linestyle="--")
        plt.savefig(outdir / "plots" / "searched_area.png")

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
        plt.savefig(outdir / "plots" / "fifty_ninety_areas.png")
        plt.close()

        # searched prob pp-plot
        searched_probs = [
            result.searched_prob for result in crossmatch_results
        ]

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="pp_plot")
        plt.rcParams.update({"font.size": 16})

        number_of_samples = len(searched_probs)
        alphas = [0.68, 0.95, 0.997]
        for alpha in alphas:
            ax.add_confidence_band(
                number_of_samples,
                alpha=alpha,
                color=(0, 0, 0, 0.1),
                edgecolor=(0, 0, 0, 0.2),
                annotate=False,
            )
        ax.add_diagonal()
        p = scipy.stats.kstest(searched_probs, "uniform").pvalue
        ax.add_series(
            searched_probs,
            label="AMPLFI"
            + r"$~({0:#.2g})$ ".format(round(p, 2))
            + str(len(searched_probs))
            + " events",
            color="saddlebrown",
            linewidth=2,
        )

        plt.title("Searched Area Probability-Probability Plot")
        ax.set_xlabel("Credible interval")
        ax.set_ylabel("Fraction of events in credible interval")
        ax.grid(True)
        ax.legend()
        fig.savefig(outdir / "plots" / "searched_prob_pp_plot.png")
        plt.close()

        # searched prob-vol pp-plot
        searched_prob_vols = [
            result.searched_prob_vol for result in crossmatch_results
        ]

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="pp_plot")
        plt.rcParams.update({"font.size": 16})

        number_of_samples = len(searched_prob_vols)
        alphas = [0.68, 0.95, 0.997]
        for alpha in alphas:
            ax.add_confidence_band(
                number_of_samples,
                alpha=alpha,
                color=(0, 0, 0, 0.1),
                edgecolor=(0, 0, 0, 0.2),
                annotate=False,
            )
        ax.add_diagonal()
        p = scipy.stats.kstest(searched_prob_vols, "uniform").pvalue
        ax.add_series(
            searched_prob_vols,
            label="AMPLFI"
            + r"$~({0:#.2g})$ ".format(round(p, 2))
            + str(len(searched_prob_vols))
            + " events",
            color="saddlebrown",
            linewidth=2,
        )

        plt.title("Searched Volume Probability-Probability Plot")
        ax.set_xlabel("Credible interval")
        ax.set_ylabel("Fraction of events in credible interval")
        ax.grid(True)
        ax.legend()
        fig.savefig(outdir / "plots" / "searched_prob_vol_pp_plot.png")
        plt.close()


class SaveInjectionParameters(pl.Callback):
    """
    Save injection parameters to an hdf5 file.
    """

    def on_test_epoch_end(self, trainer, pl_module: "FlowModel"):
        outdir = pl_module.test_outdir
        results = pl_module.test_results
        test_parameters: pd.DataFrame = trainer.datamodule.test_parameters
        # try to get snrs; Some testing datasets
        # (e.g. base FlowDataset and ParameterTestingDataset)
        # which generate injections on the fly calculate snrs
        # while others (e.g. StrainTestingDataset) don't
        try:
            test_parameters["snr"] = np.array(
                [result.injection_parameters["snr"] for result in results]
            )
        except KeyError:
            pass

        # for FlowDataset, extrinsic parameters
        # are randomly sampled, so append them to dataframe here
        # TODO: hacky, but more robust solutions would require refactoring.
        try:
            for key in ["dec", "psi", "phi"]:
                test_parameters[key] = trainer.datamodule.test_extrinsic[key]
        except AttributeError:
            pass
        test_parameters.to_hdf(outdir / "parameters.hdf5", key="parameters")


class EstimateSamplingLatency(pl.Callback):
    """
    Estimate the sampling latency of the model by running
    a forward pass on a batch of data and measuring the time taken.
    """

    def __init__(self, num_samples: int = 20000, num_trials: int = 10):
        self.num_samples = num_samples
        self.num_trials = num_trials
        self.logger = logging.getLogger("EstimateSamplingLatency")

    def estimate_sampling_latency(self, trainer, pl_module):
        batch = next(iter(trainer.datamodule.test_dataloader()))
        batch = trainer.datamodule.transfer_batch_to_device(
            batch, pl_module.device, 0
        )
        strain, asds, _, _ = trainer.datamodule.on_after_batch_transfer(
            batch, None
        )
        context = (strain, asds)
        times = []
        self.logger.info(
            "Estimating sampling latency by sampling "
            f" {self.num_samples} samples {self.num_trials} times..."
        )
        with torch.no_grad():
            for i in range(self.num_trials):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                _ = pl_module.model.sample(
                    self.num_samples,
                    context,
                )
                end_time.record()

                # wait for the events to be recorded
                torch.cuda.synchronize()

                # convert to seconds
                elapsed_time = start_time.elapsed_time(end_time) / 1000
                times.append(elapsed_time)
                self.logger.info(f"Trial {i + 1}: {elapsed_time:.2f} s")
        avg_time = np.mean(times)
        self.logger.info(f"Mean time: {avg_time:.2f} s")

    def on_test_start(self, trainer, pl_module):
        self.estimate_sampling_latency(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        self.estimate_sampling_latency(trainer, pl_module)
