import bilby
from . import skymap
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
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

    def to_skymap(
        self, use_distance: bool = True, adaptive: bool = True, **kwargs
    ) -> Table:
        """
        Calculate a histogram skymap from posterior samples
        The posterior dataframe and injection_parameters dict
        should have `ra` and `dec` entries.

        Args:
            use_distance:
                If `True`, estimate distance ansatz parameters
            adaptive:
                If `True`, use adaptive histogram based on
                `ligo.skymap.healpix_tree.adaptive_healpix_histogram`
            **kwargs:
                Additional arguments passed to
                `amplfi.utils.skymap.histogram_skymap`
                or `amplfi.utils.skymap.adaptive_histogram_skymap`
        """
        distance = None
        if use_distance:
            distance = self.posterior["distance"]

        func = (
            skymap.adaptive_histogram_skymap
            if adaptive
            else skymap.histogram_skymap
        )
        result = func(
            self.posterior["ra"], self.posterior["dec"], distance, **kwargs
        )
        return result

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
