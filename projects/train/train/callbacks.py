import os

import h5py
from lightning.pytorch import Callback


class SaveAugmentedBatch(Callback):
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
                f["strain"] = X.cpu().numpy()
                f["parameters"] = parameters.cpu().numpy()


class SaveAugmentedSimilarityBatch(Callback):
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

            """
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
                f["strain"] = X.cpu().numpy()
                f["parameters"] = parameters.cpu().numpy()
            """
