import os

import torch
from data import SignalDataSet
from lightning.pytorch import Trainer, callbacks, loggers
from lightning.pytorch.cli import LightningCLI

from mlpe.architectures.embeddings import ResNet
from mlpe.architectures.flows import MaskedAutoRegressiveFlow
from mlpe.injection.priors import nonspin_bbh_component_mass
from mlpe.logging import configure_logging

# def cli_main():
#     cli = LightningCLI(MaskedAutoRegressiveFlow, SignalDataSet,
#                        run=False, subclass_mode_model=False,
#                        subclass_mode_data=True)

# if __name__ == "__main__":
#     cli_main()


def main():
    background_path = os.getenv('DATA_DIR') + "/background.h5"
    ifos = ["H1", "L1"]
    batch_size = 800
    batches_per_epoch = 200
    sample_rate = 2048
    time_duration = 4
    f_max = 200
    f_min = 20
    f_ref = 40
    highpass = 25
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
    num_transforms = 30
    num_blocks = 5
    hidden_features = 100

    optimizer = torch.optim.AdamW
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
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
    priors = nonspin_bbh_component_mass()

    flow_obj = MaskedAutoRegressiveFlow(
        (param_dim, n_ifos, strain_dim),
        embedding,
        optimizer,
        scheduler,
        inference_params,
        priors,
        num_transforms=num_transforms,
        num_blocks=num_blocks,
        hidden_features=hidden_features
    )
    ckpt_path = os.getenv("BASE_DIR") + "/pl-logdir/cbc-model/version_29/checkpoints/epoch=458-step=91800.ckpt"
    #checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    #flow_obj.load_state_dict(checkpoint['state_dict'])
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
        prior=priors,
    )
    print("##### Initialized data loader, calling setup ####")
    sig_dat.setup(None)

    print("##### Dataloader initialized #####")
    torch.set_float32_matmul_precision("high")
    early_stop_cb = callbacks.EarlyStopping(
        "valid_loss", patience=30, check_finite=True, verbose=True
    )
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="epoch")
    outdir = os.getenv("BASE_DIR")
    logger = loggers.CSVLogger(save_dir=outdir + "/pl-logdir", name="cbc-model")
    print("##### Initializing trainer #####")
    trainer = Trainer(
        max_epochs=1000,
        log_every_n_steps=50,
        accelerator="cuda",
        callbacks=[early_stop_cb, lr_monitor],
        logger=logger,
        gradient_clip_val=5.0,
    )
    #trainer.fit(model=flow_obj, datamodule=sig_dat, ckpt_path=ckpt_path)
    trainer.test(model=flow_obj, datamodule=sig_dat, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()
