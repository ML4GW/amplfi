import io
import os
import shutil
from pathlib import Path

import h5py
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import s3fs
from gwpy.plot import Plot
from gwpy.timeseries import TimeSeries
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger


class SaveConfigCallback(SaveConfigCallback):
    """
    Override of `lightning.pytorch.cli.SaveConfigCallback` for use with WandB
    to ensure all the hyperparameters are logged to the WandB dashboard.
    """

    def save_config(self, trainer, _, stage):
        if stage == "fit":
            if isinstance(trainer.logger, WandbLogger):
                # pop off unecessary trainer args
                config = self.config.as_dict()
                config.pop("trainer")
                trainer.logger.experiment.config.update(self.config.as_dict())


class SaveAugmentedBatch(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            datamodule = trainer.datamodule
            device = pl_module.device

            # build and save an example training batch
            # and parameters to disk
            [X] = next(iter(trainer.train_dataloader))
            X = X.to(device)

            cross, plus, parameters = datamodule.waveform_sampler.sample(X)
            strain, asds, parameters = datamodule.inject(
                X, cross, plus, parameters
            )

            # save an example validation batch
            # and parameters to disk
            [val_cross, val_plus, val_parameters], [background] = next(
                iter(datamodule.val_dataloader())
            )
            val_cross, val_plus, val_parameters = (
                val_cross.to(device),
                val_plus.to(device),
                val_parameters.to(device),
            )
            background = background.to(device)
            keys = [
                k
                for k in datamodule.hparams.inference_params
                if k not in ["dec", "psi", "phi"]
            ]
            val_parameters = {
                k: val_parameters[:, i] for i, k in enumerate(keys)
            }
            val_strain, val_asds, val_parameters = datamodule.inject(
                background, val_cross, val_plus, val_parameters
            )

            save_dir = trainer.logger.log_dir or trainer.logger.save_dir

            if save_dir.startswith("s3://"):
                s3 = s3fs.S3FileSystem()
                with s3.open(f"{save_dir}/batch.h5", "wb") as s3_file:
                    with io.BytesIO() as f:
                        with h5py.File(f, "w") as h5file:
                            h5file["strain"] = strain.cpu().numpy()
                            h5file["asds"] = asds.cpu().numpy()
                            h5file["parameters"] = parameters.cpu().numpy()
                        s3_file.write(f.getvalue())

                with s3.open(f"{save_dir}/val-batch.h5", "wb") as s3_file:
                    with io.BytesIO() as f:
                        with h5py.File(f, "w") as h5file:
                            h5file["strain"] = val_strain.cpu().numpy()
                            h5file["asds"] = val_asds.cpu().numpy()
                            h5file["parameters"] = val_parameters.cpu().numpy()
                        s3_file.write(f.getvalue())
            else:
                with h5py.File(
                    os.path.join(save_dir, "train-batch.h5"), "w"
                ) as f:
                    f["strain"] = strain.cpu().numpy()
                    f["asds"] = asds.cpu().numpy()
                    f["parameters"] = parameters.cpu().numpy()

                with h5py.File(
                    os.path.join(save_dir, "val-batch.h5"), "w"
                ) as f:
                    f["strain"] = val_strain.cpu().numpy()
                    f["asds"] = val_asds.cpu().numpy()
                    f["parameters"] = val_parameters.cpu().numpy()


class SaveAugmentedSimilarityBatch(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            datamodule = trainer.datamodule
            device = pl_module.device
            # build and save an example training batch
            # and parameters to disk
            [X] = next(iter(trainer.train_dataloader))
            X = X.to(device)

            cross, plus, parameters = datamodule.waveform_sampler.sample(X)
            [ref, aug], asds, parameters = datamodule.inject(
                X, cross, plus, parameters
            )

            save_dir = trainer.logger.log_dir or trainer.logger.save_dir

            with h5py.File(os.path.join(save_dir, "train-batch.h5"), "w") as f:
                f["ref"] = ref.cpu().numpy()
                f["aug"] = aug.cpu().numpy()
                f["asds"] = asds.cpu().numpy()
                f["parameters"] = parameters.cpu().numpy()

            # save an example validation batch
            # and parameters to disk
            [cross, plus, parameters], [background] = next(
                iter(datamodule.val_dataloader())
            )
            cross, plus, parameters = (
                cross.to(device),
                plus.to(device),
                parameters.to(device),
            )
            background = background.to(device)
            keys = [
                k
                for k in datamodule.hparams.inference_params
                if k not in ["dec", "psi", "phi"]
            ]
            parameters = {k: parameters[:, i] for i, k in enumerate(keys)}
            [ref, aug], asds, parameters = datamodule.inject(
                background, cross, plus, parameters
            )
            with h5py.File(os.path.join(save_dir, "val-batch.h5"), "w") as f:
                f["ref"] = ref.cpu().numpy()
                f["aug"] = aug.cpu().numpy()
                f["asds"] = asds.cpu().numpy()
                f["parameters"] = parameters.cpu().numpy()


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_train_end(self, trainer, pl_module):
        save_dir = trainer.logger.save_dir
        shutil.copy(self.best_model_path, os.path.join(save_dir, "best.ckpt"))


class StrainVisualization(pl.Callback):
    """
    Lightning Callback for visualizing the strain data
    and asds being analyzed during the test step
    """

    def __init__(self, outdir: Path, num_plot: int):
        self.outdir = outdir
        self.num_plot = num_plot

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

        outdir = self.outdir / f"event_{batch_idx}"
        outdir.mkdir(exist_ok=True)

        # unpack batch
        strain, asds, _ = batch
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
            geometry=(1, 2),
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

        axes[0].set_title("Real")
        axes[1].set_title("Imaginary")

        axes[0].set_ylabel("Whitened Amplitude")
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].set_xlabel("Frequency (Hz)")

        plt.legend()

        plt.savefig(whitened_fd_strain_fname)
        plt.close()

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        return self.on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )


class SavePosteriors(pl.Callback):
    """
    Lightning Callback for bilby result objects and
    posterior data to disk
    """

    def __init__(self, outdir: Path):
        self.outdir = outdir

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

        outdir = self.outdir / f"event_{batch_idx}"
        outdir.mkdir(exist_ok=True)

        # save posterior samples for ease of use with
        # ligo skymap and save full result to have
        # access to the true injection parameters
        result.save_posterior_samples(outdir / "posterior_samples.dat")
        result.save_to_file(outdir / "result.hdf5", extension="hdf5")

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        return self.on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
