from typing import Callable, Tuple

import torch
import torch.distributions as dist
from lightning import pytorch as pl
from pyro.distributions.conditional import ConditionalComposeTransformModule
from pyro.distributions.transforms import ConditionalAffineCoupling
from pyro.nn import ConditionalDenseNN

from mlpe.architectures.flows import utils
from mlpe.architectures.flows.flow import NormalizingFlow
from mlpe.data.transforms import Preprocessor


class CouplingFlow(pl.LightningModule, NormalizingFlow):
    def __init__(
        self,
        shape: Tuple[int, int, int],
        embedding_net: torch.nn.Module,
        preprocessor: Preprocessor,
        opt: torch.optim.SGD,
        sched: torch.optim.lr_scheduler.ConstantLR,
        inference_params: list,
        priors: dict,
        num_samples_draw: int = 3000,
        num_plot_corner: int = 20,
        hidden_features: int = 512,
        num_transforms: int = 5,
        num_blocks: int = 2,
        dropout_probability: float = 0.0,
        activation: Callable = torch.relu,
    ):
        super().__init__()
        self.param_dim, self.n_ifos, self.strain_dim = shape
        self.split_dim = self.param_dim // 2
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.num_transforms = num_transforms
        self.activation = activation
        self.optimizer = opt
        self.scheduler = sched
        self.priors = priors
        self.inference_params = inference_params
        self.num_samples_draw = num_samples_draw
        self.num_plot_corner = num_plot_corner
        # define embedding net and base distribution
        self.embedding_net = embedding_net
        self.preprocessor = preprocessor
        # don't train preprocessor
        for n, p in self.preprocessor.named_parameters():
            p.required_grad = False
        # build the transform - sets the transforms attrib
        self.build_flow()

    def transform_block(self):
        """Returns single affine coupling transform"""
        arn = ConditionalDenseNN(
            self.split_dim,
            self.context_dim,
            [self.hidden_features],
            param_dims=[
                self.param_dim - self.split_dim,
                self.param_dim - self.split_dim,
            ],
            nonlinearity=self.activation,
        )
        transform = ConditionalAffineCoupling(self.split_dim, arn)
        return transform

    def distribution(self):
        """Returns the base distribution for the flow"""
        return dist.Normal(
            torch.zeros(self.param_dim, device=self.device),
            torch.ones(self.param_dim, device=self.device),
        )

    def build_flow(self):
        self.transforms = []
        for _ in range(self.num_transforms):
            self.transforms.extend([self.transform_block()])

        self.transforms = ConditionalComposeTransformModule(self.transforms)

    def training_step(self, batch, batch_idx):
        strain, parameters = batch
        strain, parameters = self.preprocessor(strain, parameters)
        loss = -self.log_prob(parameters, context=strain).mean()
        self.log(
            "train_loss", loss, on_step=True, prog_bar=True, sync_dist=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        strain, parameters = batch
        strain, parameters = self.preprocessor(strain, parameters)
        loss = -self.log_prob(parameters, context=strain).mean()
        self.log(
            "valid_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def on_test_epoch_start(self):
        self.test_results = []
        self.num_plotted = 0

    def test_step(self, batch, batch_idx):
        strain, parameters = batch
        res = utils.draw_samples_from_model(
            strain,
            parameters,
            self,
            self.preprocessor,
            self.inference_params,
            self.num_samples_draw,
            self.priors,
        )
        self.test_results.append(res)
        if batch_idx % 100 == 0 and self.num_plotted < self.num_plot_corner:
            skymap_filename = f"{self.num_plotted}_mollview.png"
            res.plot_corner(
                save=True,
                filename=f"{self.num_plotted}_corner.png",
                levels=(0.5, 0.9),
            )
            utils.plot_mollview(
                res.posterior["phi"],
                res.posterior["dec"],
                truth=(
                    res.injection_parameters["phi"],
                    res.injection_parameters["dec"],
                ),
                outpath=skymap_filename,
            )
            self.num_plotted += 1
            self.print("Made corner plots and skymap for ", batch_idx)

    def on_test_epoch_end(self):
        import bilby

        bilby.result.make_pp_plot(
            self.test_results,
            save=True,
            filename="pp-plot.png",
            keys=self.inference_params,
        )
        del self.test_results, self.num_plotted

    def configure_optimizers(self):
        opt = self.optimizer(self.parameters())
        sched = self.scheduler(opt)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched}}
