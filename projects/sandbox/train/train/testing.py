from pathlib import Path
from typing import Optional, Tuple

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


def plot_mollview(
    ra_samples: np.ndarray,
    dec_samples: np.ndarray,
    nside: int = 32,
    truth: Optional[Tuple[float, float]] = None,
    outpath: Path = None,
):
    """Plot mollview of posterior samples

    Args:
        ra_samples: array of right ascension samples in radians (-pi, pi)
        dec_samples: array of declination samples in radians (-pi/2, pi/2)
        nside: nside parameter for healpy
        truth: tuple of true ra and dec
    """

    # mask out non physical samples;
    ra_samples_mask = (ra_samples > -np.pi) * (ra_samples < np.pi)
    dec_samples += np.pi / 2
    dec_samples_mask = (dec_samples > 0) * (dec_samples < np.pi)

    net_mask = ra_samples_mask * dec_samples_mask
    ra_samples = ra_samples[net_mask]
    dec_samples = dec_samples[net_mask]

    # calculate number of samples in each pixel
    NPIX = hp.nside2npix(nside)
    ipix = hp.ang2pix(nside, dec_samples, ra_samples)
    ipix = np.sort(ipix)
    uniq, counts = np.unique(ipix, return_counts=True)

    # create empty map and then fill in non-zero pix with counts
    m = np.zeros(NPIX)
    m[np.in1d(range(NPIX), uniq)] = counts

    plt.close()
    # plot molleweide
    fig = hp.mollview(m)
    if truth is not None:
        ra_inj, dec_inj = truth
        dec_inj += np.pi / 2
        hp.visufunc.projscatter(
            dec_inj, ra_inj, marker="x", color="red", s=150
        )

    plt.savefig(outpath)

    return fig
