from pathlib import Path

import bilby
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


class Result(bilby.result.Result):
    def get_sky_projection(
        self,
        nside: int = 32,
    ):
        """Store a HEALPix array with the sky coordinates

        Args:
            nside: nside parameter for healpy
        """
        ra = self.posterior["phi"]
        dec = self.posterior["dec"]
        theta = np.pi / 2 - dec

        # mask out non physical samples;
        mask = (ra > -np.pi) * (ra < np.pi)
        mask &= (theta > 0) * (theta < np.pi)

        ra = ra[mask]
        dec = dec[mask]
        theta = theta[mask]

        # calculate number of samples in each pixel
        NPIX = hp.nside2npix(nside)
        ipix = hp.ang2pix(nside, theta, ra)
        ipix = np.sort(ipix)
        uniq, counts = np.unique(ipix, return_counts=True)

        # create empty map and then fill in non-zero pix with counts
        m = np.zeros(NPIX)
        m[np.in1d(range(NPIX), uniq)] = counts

        return m

    def calculate_searched_area(self, nside: int = 32):
        healpix = self.get_sky_projection(nside=nside)

        ra_inj = self.injection_parameters["phi"]
        dec_inj = self.injection_parameters["dec"]
        theta_inj = np.pi / 2 - dec_inj
        true_ipix = hp.ang2pix(nside, theta_inj, ra_inj)

        sorted_idxs = np.argsort(healpix)[
            ::-1
        ]  # sort pixels in descending order
        # count number of pixels before hitting the pixel with injection
        # in the sorted array
        num_pix_before_injection = 1 + np.argmax(sorted_idxs == true_ipix)
        searched_area = num_pix_before_injection * hp.nside2pixarea(
            nside, degrees=True
        )
        return searched_area

    def plot_mollview(
        self,
        outpath: Path = None,
        nside: int = 32,
    ):
        healpix = self.get_sky_projection(nside)
        ra_inj = self.injection_parameters["phi"]
        dec_inj = self.injection_parameters["dec"]
        theta_inj = np.pi / 2 - dec_inj
        plt.close()
        # plot molleweide
        fig = hp.mollview(healpix)
        hp.visufunc.projscatter(
            theta_inj, ra_inj, marker="x", color="red", s=150
        )

        plt.savefig(outpath)

        return fig
