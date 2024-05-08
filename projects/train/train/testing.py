from pathlib import Path

import bilby
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


class Result(bilby.result.Result):
    def plot_mollview(
        self,
        nside: int = 32,
        outpath: Path = None,
    ):
        """Plot mollview of sky localization posterior samples

        Args:
            nside: nside parameter for healpy
            outpath: path to save the plot
        """

        ra = self.posterior["phi"]
        dec = self.posterior["dec"]
        dec += np.pi / 2

        # mask out non physical samples;
        mask = (ra > -np.pi) * (ra < np.pi)
        mask &= (dec > 0) * (dec < np.pi)

        ra = ra[mask]
        dec = dec[mask]

        # calculate number of samples in each pixel
        NPIX = hp.nside2npix(nside)
        ipix = hp.ang2pix(nside, dec, ra)
        ipix = np.sort(ipix)
        uniq, counts = np.unique(ipix, return_counts=True)

        # create empty map and then fill in non-zero pix with counts
        m = np.zeros(NPIX)
        m[np.in1d(range(NPIX), uniq)] = counts

        plt.close()
        # plot molleweide
        fig = hp.mollview(m)

        ra_inj = self.injection_parameters["phi"]
        dec_inj = self.injection_parameters["dec"]
        dec_inj += np.pi / 2
        hp.visufunc.projscatter(
            dec_inj, ra_inj, marker="x", color="red", s=150
        )

        plt.savefig(outpath)

        return fig
