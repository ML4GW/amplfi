import bilby
from . import skymap
from typing import Optional
from pathlib import Path
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
from ligo.skymap.postprocess.crossmatch import crossmatch, CrossmatchResult
from copy import copy
import pandas as pd


class AmplfiResult(bilby.result.Result):
    """
    A subclass of `bilby.result.Result` with additional
    convenience methods for generating AMPLFI skymaps
    """

    def to_crossmatch_result(
        self,
        use_distance: bool = True,
        min_samples_per_pix_dist: int = 5,
        **kwargs,
    ) -> CrossmatchResult:
        """
        Calculate a `ligo.skymap.postprocess.crossmatch.CrossmatchResult`
        based on sky localization and distance posterior samples.
        The posterior dataframe and injection_parameters dict
        should have `ra` and `dec` entries

        """
        skymap = self.to_skymap(
            use_distance=use_distance,
            min_samples_per_pix_dist=min_samples_per_pix_dist,
        )

        coordinates = SkyCoord(
            self.injection_parameters["ra"] * u.rad,
            self.injection_parameters["dec"] * u.rad,
            distance=self.injection_parameters["distance"] * u.Mpc,
        )
        return crossmatch(skymap, coordinates, contours=(50, 90))

    def to_skymap(self, use_distance: bool = True, **kwargs) -> Table:
        """
        Calculate a histogram skymap from posterior samples
        The posterior dataframe and injection_parameters dict
        should have `ra` and `dec` entries

        Args:
            use_distance:
                If `True`, estimate distance ansatz parameters
            **kwargs:
                Additional arguments passed to
                `amplfi.utils.skymap.histogram_skymap`
        """
        distance = None
        if use_distance:
            distance = self.posterior["distance"]

        return skymap.histogram_skymap(
            self.posterior["ra"], self.posterior["dec"], distance, **kwargs
        )

    def plot_skymap(
        self, outpath: Optional[Path] = None, **kwargs
    ) -> plt.Figure:
        """
        Plot a mollweide projection of the skymap.

        Expected that the `self.posterior` pandas dataframe
        `self.injection_parameters` dictionary have `
        ra` and `dec` entries.

        Args:
            outpath:
                Optional file path to save skymap
            **kwargs:
                Additional kwargs passed to `AmplfiResult.to_skymap`
        """

        skymap = self.to_skymap(**kwargs)

        ax = plt.figure().add_subplot(projection="astro mollweide")
        ax.grid()
        ra_inj = self.injection_parameters["phi"]
        dec_inj = self.injection_parameters["dec"]
        ax.plot_coord(
            SkyCoord(ra_inj, dec_inj, unit=u.rad),
            "x",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=5,
        )
        sr_to_deg2 = u.sr.to(u.deg**2)
        skymap["PROBDENSITY"] *= 1 / sr_to_deg2
        ax.imshow_hpx(
            (skymap, "ICRS"), vmin=0, order="nearest-neighbor", cmap="cylon"
        )
        plt.savefig(outpath)

    def reweight_to_prior(
        self,
        target_prior: bilby.core.prior.PriorDict,
        max_rejection_steps: int = 1,
    ) -> "AmplfiResult":
        reweighted_result = copy(self)
        keys = list(target_prior.keys())
        samples_dict = {key: self.posterior[key].values for key in keys}
        target_probs = target_prior.ln_prob(samples_dict, axis=0)
        ln_weights = target_probs - self.posterior["log_prior"].values
        weights = np.exp(ln_weights)

        num_samples = len(self.posterior)
        reweighted_posterior = pd.DataFrame(columns=self.posterior.columns)
        for _ in range(max_rejection_steps):
            if len(reweighted_posterior) >= num_samples:
                break

            samples = bilby.core.result.rejection_sample(
                self.posterior, weights
            )
            reweighted_posterior = pd.concat(
                [reweighted_posterior, samples], ignore_index=True
            )

        reweighted_result.posterior = reweighted_posterior[:num_samples]
        return reweighted_result
