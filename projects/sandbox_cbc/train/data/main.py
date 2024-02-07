import os

import torch
from data import SignalDataSet
from lightning.pytorch import Trainer, callbacks, loggers
from lightning.pytorch.cli import LightningCLI

from mlpe.architectures.embeddings import ResNet
from mlpe.architectures.flows import MaskedAutoRegressiveFlow
from mlpe.injection.priors import nonspin_bbh
from mlpe.logging import configure_logging

# def cli_main():
#     cli = LightningCLI(MaskedAutoRegressiveFlow, SignalDataSet,
#                        run=False, subclass_mode_model=False,
#                        subclass_mode_data=True)

# if __name__ == "__main__":
#     cli_main()


def main():
    background_path = "/tmp/background.h5"
    ifos = ["H1", "L1"]
    batch_size = 1000
    batches_per_epoch = 200
    sample_rate = 2048
    time_duration = 4
    f_max = 200
    f_min = 20
    f_ref = 40
    highpass = 32
    valid_frac = 0.1
    learning_rate = 1e-3
    resnet_context_dim = 128
    resnet_layers = [2, 2]
    resnet_norm_groups = 8
    inference_params = [
        "chirp_mass",
        "mass_ratio",
        "luminosity_distance",
        "theta_jn",
        "dec",
        "psi",
        "phi",
    ]

    optimizer = torch.optim.AdamW(lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    param_dim, n_ifos, strain_dim = (
        len(inference_params),
        len(ifos),
        int(sample_rate * time_duration),
    )

    # models
    embedding = ResNet(
        (n_ifos, strain_dim),
        context_dim=resnet_context_dim,
        layers=resnet_layers,
        norm_groups=resnet_norm_groups,
    )
    priors = nonspin_bbh()

    flow_obj = MaskedAutoRegressiveFlow(
        (param_dim, n_ifos, strain_dim),
        embedding,
        optimizer,
        scheduler,
        inference_params,
        priors,
    )
    # data
    sig_dat = SignalDataSet(
        background_path,
        ifos,
        valid_frac,
        batch_size,
        batches_per_epoch,
        sample_rate,
        time_duration,
        f_min,
        f_max,
        f_ref,
        priors,
    )
    sig_dat.setup(None)

    torch.set_float32_matmul_precision("high")
    early_stop_cb = callbacks.EarlyStopping(
        "valid_loss", patience=30, check_finite=True, verbose=True
    )
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="epoch")
    outdir = os.getenv("PE_BASEDIR")
    logger = loggers.CSVLogger(save_dir=outdir / "pl-logdir", name="cbc-model")
    trainer = Trainer(
        max_epochs=300,
        log_every_n_steps=50,
        accelerator="cuda",
        callbacks=[early_stop_cb, lr_monitor],
        logger=logger,
        gradient_clip_val=5.0,
    )
    trainer.fit(flow_obj, sig_dat)
