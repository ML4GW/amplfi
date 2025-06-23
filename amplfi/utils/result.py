import bilby
from . import skymap
from typing import Optional
from pathlib import Path
import healpy as hp
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
        nside: int,
        min_samples_per_pix: int = 5,
        use_distance: bool = True,
        **crossmatch_kwargs,
    ) -> CrossmatchResult:
        """
        Calculate a `ligo.skymap.postprocess.crossmatch.CrossmatchResult`
        based on sky localization and distance posterior samples.
        The posterior dataframe and injection_parameters dict
        should have `ra` and `dec` entries

        """
        skymap = self.to_skymap(
            nside=nside,
            min_samples_per_pix=min_samples_per_pix,
            use_distance=use_distance,
        )

        coordinates = SkyCoord(
            self.injection_parameters["ra"] * u.rad,
            self.injection_parameters["dec"] * u.rad,
            distance=self.injection_parameters["distance"] * u.Mpc,
        )
        return crossmatch(skymap, coordinates, **crossmatch_kwargs)

    def to_skymap(
        self,
        nside: int,
        min_samples_per_pix: int = 5,
        use_distance: bool = True,
    ) -> Table:
        """
        Calculate a histogram skymap from posterior samples
        The posterior dataframe and injection_parameters dict
           should have `ra` and `dec` entries
        """
        distance = None
        if use_distance:
            distance = self.posterior["distance"]

        return skymap.histogram_skymap(
            self.posterior["ra"],
            self.posterior["dec"],
            distance,
            nside=nside,
            min_samples_per_pix=min_samples_per_pix,
        )

    def calculate_searched_area(
        self, nside: int = 32
    ) -> tuple[float, float, float]:
        """
        Calculate the searched area, and estimates
        of 50% and 90% credible region.
        The posterior dataframe and injection_parameters dict
        should have `ra` and `dec` entries
        """
        smap = self.to_skymap(nside)["PROBDENSITY"]

        return skymap.calculate_searched_area(
            smap,
            self.injection_parameters["ra"],
            self.injection_parameters["dec"],
            nside=nside,
        )

    def plot_mollview(
        self, nside: int = 32, outpath: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot a mollweide projection of the skymap.
        The posterior dataframe and injection_parameters dict
        should have `ra` and `dec` entries
        """
        skymap = self.to_skymap(nside)["PROBDENSITY"]

        # plot molleweide
        plt.close()
        fig = hp.mollview(skymap, nest=True)

        # plot true values if available
        if self.injection_parameters is not None:
            ra_inj = self.injection_parameters["phi"]
            dec_inj = self.injection_parameters["dec"]
            theta_inj = np.pi / 2 - dec_inj

            hp.visufunc.projscatter(
                theta_inj, ra_inj, marker="x", color="red", s=150
            )

        plt.savefig(outpath)

        return fig

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
