from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from astropy import io
from pathlib import Path
from typing import TYPE_CHECKING
from gwpy.plot import Plot
from gwpy.timeseries import TimeSeries
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import h5py
import bilby
from tqdm.auto import tqdm
import ligo.skymap.plot  # noqa: F401

if TYPE_CHECKING:
    from ligo.skymap.postprocess.crossmatch import CrossmatchResult
    from amplfi.train.models import FlowModel
    from amplfi.utils.result import AmplfiResult


class StrainVisualization(pl.Callback):
    """
    Lightning Callback for visualizing the strain data
    and asds being analyzed during the test step
    """

    def __init__(self, outdir: Path, num_plot: int, save_data: bool = True):
        self.outdir = outdir
        self.num_plot = num_plot
        self.save_data = save_data

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        """
        Called at the end of each test step.
        `outputs` consists of objects returned by `pl_module.test_step`.
        """

        # test_step returns bilby result object
        result = outputs

        if batch_idx >= self.num_plot:
            return

        outdir = self.outdir / str(batch_idx)
        outdir.mkdir(exist_ok=True)

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

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        return self.on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )


class PlotMollview(pl.Callback):
    """
    Lightning Callback for plotting mollview skymaps
    after each test batch
    """

    def __init__(self, outdir: Path, nside: int):
        self.outdir = outdir
        self.nside = nside

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

        # test_step returns bilby result object
        result = outputs

        outdir = self.outdir / str(batch_idx)
        outdir.mkdir(exist_ok=True)
        skymap_filename = outdir / "mollview.png"
        result.plot_mollview(
            self.nside,
            outpath=skymap_filename,
        )

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        return self.on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )


class PlotCorner(pl.Callback):
    """
    Lightning Callback for making corner plots after test epoch
    """

    def __init__(self, outdir: Path):
        self.outdir = outdir

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

        # test_step returns bilby result object
        result = outputs
        outdir = self.outdir / str(batch_idx)
        self.plot_corner(result, outdir)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # test_step returns bilby result object
        result = outputs

        # for predict step, use gpstime to name the directory
        gpstime = batch[2].cpu().numpy()[0]
        outdir = self.outdir / f"event_{int(gpstime)}"
        self.plot_corner(result, outdir)


class SaveFITS(pl.Callback):
    """ """

    def __init__(self, outdir: Path, nside: int, min_samples_per_pix: int):
        self.outdir = outdir
        self.nside = nside
        self.min_samples_per_pix = min_samples_per_pix

    def save_fits(
        self,
        result: "AmplfiResult",
        outdir: Path,
    ):
        fits = io.fits.table_to_hdu(
            result.to_skymap(
                self.nside,
                self.min_samples_per_pix,
                use_distance=True,
            )
        )
        outdir.mkdir(exist_ok=True)
        fits.writeto(outdir / "amplfi.skymap.fits", overwrite=True)

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

        # test_step returns bilby result object
        result = outputs
        outdir = self.outdir / str(batch_idx)
        outdir.mkdir(exist_ok=True)
        self.save_fits(result, outdir)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # test_step returns bilby result object
        result = outputs

        # for predict step, use gpstime to name the directory
        gpstime = batch[2].cpu().numpy()[0]
        outdir = self.outdir / f"event_{int(gpstime)}"
        self.save_fits(result, outdir)


class SavePosterior(pl.Callback):
    """
    Lightning Callback for bilby result objects and
    posterior data to disk
    """

    def __init__(self, outdir: Path):
        self.outdir = outdir

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

        # test_step returns bilby result object
        result = outputs

        outdir = self.outdir / str(batch_idx)
        outdir.mkdir(exist_ok=True)

        self.save_posterior(result, outdir)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # test_step returns bilby result object
        result = outputs

        # for predict step, use gpstime to name the directory
        gpstime = batch[2].cpu().numpy()[0]
        outdir = self.outdir / f"event_{int(gpstime)}"
        outdir.mkdir(exist_ok=True)

        self.save_posterior(result, outdir)


class ProbProbPlot(pl.Callback):
    def on_test_epoch_end(self, _, pl_module: "FlowModel"):
        logger = pl_module._logger
        logger.info("Making PP plot")
        bilby.result.make_pp_plot(
            pl_module.test_results,
            save=True,
            filename=pl_module.test_outdir / "plots" / "pp-plot.png",
            keys=pl_module.inference_params,
        )


def crossmatch_skymap(
    result: "AmplfiResult",
    nside: int,
    min_samples_per_pix: int,
    contours: tuple[float],
):
    """
    Function to process each skymap and calculate crossmatch statistics.
    """

    crossmatch_result = result.to_crossmatch_result(
        nside=nside,
        min_samples_per_pix=min_samples_per_pix,
        use_distance=True,
        contours=contours,
    )
    return crossmatch_result


class CrossMatchStatistics(pl.Callback):
    """
    Use `ligo.skymap.postprocess.crossmatch` to calculate skymap statistics.
    Make searched area and searched volume CDF plots
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

    def __init__(self, contours: tuple[float] = (0.5, 0.9)):
        self.contours = contours

    def write_skymap_statistics(
        self, outdir: Path, results: list["CrossmatchResult"]
    ):
        """
        Write skymap statistics to file
        """

        with h5py.File(outdir / "skymap_stats.hdf5", "w") as f:
            for attr in self.crossmatch_attributes:
                if attr == "contour_areas":
                    for i, contour in enumerate(self.contours):
                        f.create_dataset(
                            f"contour_areas_{int(contour * 100)}",
                            data=np.array(
                                [result.contour_areas[i] for result in results]
                            ),
                        )
                else:
                    f.create_dataset(
                        attr,
                        data=np.array(
                            [getattr(result, attr) for result in results]
                        ),
                    )

    def on_test_epoch_end(self, _, pl_module: "FlowModel"):
        crossmatch_results: list["CrossmatchResult"] = []
        test_outdir = pl_module.test_outdir
        test_results = pl_module.test_results

        func = partial(
            crossmatch_skymap,
            nside=pl_module.nside,
            min_samples_per_pix=pl_module.min_samples_per_pix,
            contours=self.contours,
        )

        crossmatch_results = [None] * len(test_results)
        num_processes = min(mp.cpu_count(), len(test_results))
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_index = {
                executor.submit(func, result): idx
                for idx, result in enumerate(test_results)
            }
            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc="Crossmatching skymaps",
            ):
                idx = future_to_index[future]
                crossmatch_results[idx] = future.result()

        self.write_skymap_statistics(test_outdir, crossmatch_results)

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
        plt.savefig(test_outdir / "plots" / "searched_volume.png")

        # searched area cum hist
        plt.figure(figsize=(10, 6))
        plt.step(searched_areas, counts, where="post")
        plt.xscale("log")
        plt.xlabel("Searched Area (deg^2)")
        plt.ylabel("Cumulative Probability")
        plt.title("Searched Area Cumulative Distribution Function")
        plt.grid()
        plt.axhline(0.5, color="grey", linestyle="--")
        plt.savefig(test_outdir / "plots" / "searched_area.png")

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
        plt.savefig(test_outdir / "plots" / "fifty_ninety_areas.png")
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
        fig.savefig(test_outdir / "plots" / "searched_prob_pp_plot.png")
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
        fig.savefig(test_outdir / "plots" / "searched_prob_vol_pp_plot.png")
        plt.close()


class SaveInjectionParameters(pl.Callback):
    """
    Save randomly sampled injection parameters to a file
    at the end of each test epoch.
    """

    def __init__(self, outdir: Path):
        self.outdir = outdir

    # TODO: should these parameters be saved
    # regardless of whether this callback is activated?
    # i.e. should `on_test_epoch_start` and `on_test_batch_end`
    # be moved to the base dataset?
    def on_test_epoch_start(self, trainer, pl_module: "FlowModel"):
        # initialize field in test parameters for snr
        num_test = len(trainer.datamodule.test_dataloader())
        for param in ["snr", "dec", "psi", "phi", "ra"]:
            trainer.datamodule.test_parameters[param] = np.zeros(num_test)

    def on_test_epoch_end(self, trainer, pl_module: "FlowModel"):
        # save parameters of randomly sampled injections
        with h5py.File(self.outdir / "parameters.hdf5", "w") as f:
            for param, data in trainer.datamodule.test_parameters.items():
                f.create_dataset(param, data=data)

    def on_test_batch_end(
        self,
        trainer,
        pl_module: "FlowModel",
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        result: AmplfiResult = outputs
        for param in ["snr", "dec", "psi", "phi", "ra"]:
            trainer.datamodule.test_parameters[param][batch_idx] = (
                result.injection_parameters[param]
            )
