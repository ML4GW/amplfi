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


class AmplfiResult(bilby.result.Result):
    """
    A subclass of `bilby.result.Result` with additional
    convenience methods for generating AMPLFI skymaps
    """

    def to_crossmatch_result(
        self,
        nside: int,
        min_samples_per_pix: int = 15,
        use_distance: bool = True,
        **crossmatch_kwargs,
    ) -> CrossmatchResult:
        """
        Calculate a `ligo.skymap.postprocess.crossmatch.CrossmatchResult`
        based on sky localization and distance posterior samples
        """
        skymap = self.to_skymap(
            nside=nside,
            min_samples_per_pix=min_samples_per_pix,
            use_distance=use_distance,
        )

        coordinates = SkyCoord(
            self.injection_parameters["phi"] * u.rad,
            self.injection_parameters["dec"] * u.rad,
            distance=self.injection_parameters["distance"] * u.Mpc,
        )
        return crossmatch(skymap, coordinates, **crossmatch_kwargs)

    def to_skymap(
        self,
        nside: int,
        min_samples_per_pix: int = 15,
        use_distance: bool = True,
    ) -> Table:
        """Calculate a histogram skymap from posterior samples"""
        distance = None
        if use_distance:
            distance = self.posterior["distance"]

        return skymap.histogram_skymap(
            self.posterior["phi"],
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
        of 50% and 90% credible region
        """
        skymap = self.to_skymap(nside)["PROBDENSITY"]

        return skymap.calculate_searched_area(
            skymap,
            self.injection_parameters["phi"],
            self.injection_parameters["dec"],
            nside=nside,
        )

    def plot_mollview(
        self, nside: int = 32, outpath: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot a mollweide projection of the skymap
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
