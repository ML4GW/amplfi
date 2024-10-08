import os
import shutil

import h5py
import lightning.pytorch as pl
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
            strain, parameters = datamodule.inject(X, cross, plus, parameters)

            save_dir = trainer.logger.log_dir or trainer.logger.save_dir
            with h5py.File(os.path.join(save_dir, "train-batch.h5"), "w") as f:
                f["strain"] = strain.cpu().numpy()
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
            strain, parameters = datamodule.inject(
                background, cross, plus, parameters
            )
            with h5py.File(os.path.join(save_dir, "val-batch.h5"), "w") as f:
                f["strain"] = strain.cpu().numpy()
                f["parameters"] = parameters.cpu().numpy()


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
            [ref, aug], parameters = datamodule.inject(
                X, cross, plus, parameters
            )

            save_dir = trainer.logger.log_dir or trainer.logger.save_dir
            with h5py.File(os.path.join(save_dir, "train-batch.h5"), "w") as f:
                f["ref"] = ref.cpu().numpy()
                f["aug"] = aug.cpu().numpy()
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
            [ref, aug], parameters = datamodule.inject(
                background, cross, plus, parameters
            )
            with h5py.File(os.path.join(save_dir, "val-batch.h5"), "w") as f:
                f["ref"] = ref.cpu().numpy()
                f["aug"] = aug.cpu().numpy()
                f["parameters"] = parameters.cpu().numpy()


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_train_end(self, trainer, pl_module):
        save_dir = trainer.logger.save_dir
        shutil.copy(self.best_model_path, os.path.join(save_dir, "best.ckpt"))
