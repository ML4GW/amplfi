import astropy_healpix as ah
import bilby
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from astropy import io, table
from astropy import units as u
from mhealpy import HealpixMap

from ..architectures.flows import FlowArchitecture
from ..testing import Result
from .base import AmplfiModel

Tensor = torch.Tensor


def nest2uniq(nside, ipix):
    # return ipix + 4**(order + 1)
    return 4 * nside * nside + ipix


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
        samples_per_event: int = 200000,
        num_corner: int = 10,
        nside: int = 64,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        # construct our model
        self.model = arch
        self.samples_per_event = samples_per_event
        self.num_corner = num_corner
        self.nside = nside

        # save our hyperparameters
        self.save_hyperparameters(ignore=["arch"])

        # if checkpoint is not None, load in model weights;
        # checkpint should only be specified here if running trainer.test
        self.maybe_load_checkpoint(self.checkpoint)

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

    def get_skymap_hpx(self, samples, context, top_nside=16, rounds=8):
        """Adaptive grid implemented from ligo.skymap"""
        # calculate density of the original samples
        log_prob_orig = self.model.log_prob(samples, context=context)

        top_npix = ah.nside_to_npix(top_nside)
        nrefine = top_npix // 4
        cells = zip([0] * nrefine, [top_nside // 2] * nrefine, range(nrefine))
        for iround in range(rounds + 1):
            print(
                "adaptive refinement round {} of {} ...".format(iround, rounds)
            )
            cells = sorted(cells, key=lambda p_n_i: p_n_i[0] / p_n_i[1] ** 2)
            new_nside, new_ipix = np.transpose(
                [
                    (nside * 2, ipix * 4 + i)
                    for _, nside, ipix in cells[-nrefine:]
                    for i in range(4)
                ]
            )
            lon, lat = ah.healpix_to_lonlat(
                new_ipix, new_nside, order="nested"
            )
            n_pix = len(lon)
            lon = torch.from_numpy(lon.value).to(
                dtype=torch.float32, device=self.device
            )
            lon[lon > torch.pi] -= 2 * torch.pi
            lat = torch.from_numpy(lat.value).to(
                dtype=torch.float32, device=self.device
            )
            lon_mean = self.trainer.datamodule.scaler.mean[7]
            lon_std = self.trainer.datamodule.scaler.std[7]
            lat_mean = self.trainer.datamodule.scaler.mean[5]
            lat_std = self.trainer.datamodule.scaler.std[5]
            # scale lon, lat
            lon = (lon - lon_mean) / lon_std
            lat = (lat - lat_mean) / lat_std

            n_pix = len(lon)
            n_pts_per_pix = samples.shape[0] // n_pix
            # drop last few samples and their densities
            samples = samples[: (n_pix * n_pts_per_pix)]

            log_prob_orig = log_prob_orig[: (n_pix * n_pts_per_pix)]
            log_prob_orig = log_prob_orig.reshape(n_pts_per_pix, n_pix)

            dec_samples = lat.repeat(
                n_pts_per_pix
            )  # shape (n_pts_per_pix * n_pix)
            phi_samples = lon.repeat(n_pts_per_pix)

            # take original samples and replace the RA and Dec from pixels
            parameters = torch.vstack((
                samples[..., 0],
                samples[..., 1],
                samples[..., 2],
                samples[..., 3],
                samples[..., 4],
                dec_samples,
                samples[..., 6],
                phi_samples,
            )).mT  # shape (n_pts_per_pix * n_pix, n_dim)
            # reshape samples as (n_pts_per_pix, n_pix, n_dim)
            parameters = parameters.reshape(n_pts_per_pix, n_pix, -1)
            # store distances separately
            log_prob_new = self.model.log_prob(
                parameters, context
            )  # shape (n_pts_per_pix, n_pix)
            logw = log_prob_new - log_prob_orig

            p = torch.logsumexp(logw, dim=0)  # sum across n_pts_per_pix

            # d = torch.einsum("ij,ij->j", distances, log_prob.exp())
            # d_scaled = d * self.trainer.datamodule.scaler.std[2]
            # d_scaled += self.trainer.datamodule.scaler.mean[2]

            # d_sq = torch.einsum("ij,ij->j", distances**2, log_prob.exp())
            # d_sq_scaled = d_sq * self.trainer.datamodule.scaler.std[2] ** 2
            # d_sq_scaled += (
            #     2
            #     * self.trainer.datamodule.scaler.mean[2]
            #     * self.trainer.datamodule.scaler.std[2]
            #     * d
            # )
            # d_sq_scaled += self.trainer.datamodule.scaler.mean[2] ** 2 * p
            cells[-nrefine:] = zip(p, new_nside, new_ipix)

        logpost, nside, ipix = zip(*cells)
        logpost = torch.stack(logpost)
        nside = np.stack(nside)
        ipix = np.stack(ipix)
        uniq = nest2uniq(nside, ipix)
        logpost = logpost.cpu().numpy()
        logpost += np.log(ah.nside_to_pixel_area(nside).to_value(u.sr))
        shift = logpost.max()
        post = np.exp(logpost - shift)
        post /= post.sum()

        t = table.Table(
            [uniq, post], names=["UNIQ", "PROBDENSITY"], copy=False
        )
        m = io.fits.table_to_hdu(t)
        # helper headers for ligo.skymap cli
        m.header.extend(
            [
                ("PIXTYPE", "HEALPIX", "HEALPIX pixelisation"),
                (
                    "ORDERING",
                    "NUNIQ",
                    "Pixel ordering scheme: RING, NESTED, or NUNIQ",
                ),
            ]
        )
        return m

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
        self.test_results.append(result)

        # plot corner and skymap for a subset of the test results
        if self.idx < self.num_corner:
            skymap_filename = self.outdir / f"{self.idx}_mollview.png"
            skymap_fits_filename = self.outdir / f"{self.idx}.fits"
            skymap_fits_png = self.outdir / f"{self.idx}.fits.png"
            corner_filename = self.outdir / f"{self.idx}_corner.png"
            result.plot_corner(
                save=True,
                filename=corner_filename,
                levels=(0.5, 0.9),
            )
            result.save_posterior_samples(
                self.outdir / f"{self.idx}_samples.dat"
            )
            result.plot_mollview(
                self.nside,
                outpath=skymap_filename,
            )
            skymap_hpx = self.get_skymap_hpx(samples, context)
            # write fits file
            skymap_hpx.writeto(skymap_fits_filename, overwrite=True)
            # create a plot for the UNIQ Healpix map
            mhpx = HealpixMap(
                data=skymap_hpx.data["PROBDENSITY"],
                uniq=skymap_hpx.data["UNIQ"],
            )
            mhpx.plot()
            plt.savefig(skymap_fits_png)
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
        for result in self.test_results:
            searched_area = result.calculate_searched_area(self.nside)
            searched_areas.append(searched_area)
        searched_areas = np.sort(searched_areas)
        counts = np.arange(1, len(searched_areas) + 1) / len(searched_areas)

        plt.figure(figsize=(10, 6))
        plt.step(searched_areas, counts, where="post")
        plt.xscale("log")
        plt.xlabel("Searched Area (deg^2)")
        plt.ylabel("Cumulative Probability")
        plt.title("Searched Area Cumulative Distribution Function")
        plt.savefig(self.outdir / "searched_area.png")
        np.save(self.outdir / "searched_area.npy", searched_areas)
